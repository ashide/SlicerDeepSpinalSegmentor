import os
import unittest
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
from vtk.util import numpy_support
import vtkSegmentationCorePython as vtkSegmentationCore
try:
    from pip import main as pipmain
except:
    from pip._internal import main as pipmain
pipmain(["install", "requests"])
import requests

#
# DeepSpinalSegmentor
#


class DeepSpinalSegmentor(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Spinal Segmentor"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Shide Adibi (University of Koblenz), Sabine Bauer (University of Koblenz)"]
        self.parent.helpText = """TODO"""
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = """TODO"""  # replace with organization, grant and thanks.

#
# DeepSpinalSegmentorWidget
#


class DeepSpinalSegmentorWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Instantiate and connect widgets ...

        #
        # Parameters Area
        #
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)

        # Layout within the dummy collapsible button
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        #
        # input volume selector
        #
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        parametersFormLayout.addRow("Segmentation Input: ", self.inputSelector)

        #
        # axis selector
        #

        self.axisSelectorComboBox = qt.QComboBox()
        self.axisSelectorComboBox.addItem("X", 0)
        self.axisSelectorComboBox.addItem("Y", 1)
        self.axisSelectorComboBox.addItem("Z", 2)
        parametersFormLayout.addRow(
            "Sagital Axis: ", self.axisSelectorComboBox)

        #
        # output volume selector
        #
        self.outputSelector = slicer.qMRMLNodeComboBox()
        self.outputSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.outputSelector.selectNodeUponCreation = True
        self.outputSelector.addEnabled = True
        self.outputSelector.removeEnabled = False
        self.outputSelector.noneEnabled = True
        self.outputSelector.showHidden = False
        self.outputSelector.showChildNodeTypes = False
        self.outputSelector.setMRMLScene(slicer.mrmlScene)
        parametersFormLayout.addRow(
            "Segmentation Output: ", self.outputSelector)

        #
        # Apply Button
        #
        self.applyButton = qt.QPushButton("Start Segmentation")
        self.applyButton.enabled = False
        parametersFormLayout.addRow(self.applyButton)

        # connections
        self.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.inputSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.onSelect)
        self.outputSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.onSelect)

        # Add vertical spacer
        self.layout.addStretch(1)

        # Refresh Apply button state
        self.onSelect()

    def cleanup(self):
        pass

    def onSelect(self):
        self.applyButton.enabled = self.inputSelector.currentNode() \
            and self.outputSelector.currentNode()

    def onApplyButton(self):
        logic = DeepSpinalSegmentorLogic()
        logic.run(self.inputSelector.currentNode(),
                  self.outputSelector.currentNode(),
                  self.axisSelectorComboBox.currentText)

#
# DeepSpinalSegmentorLogic
#


class DeepSpinalSegmentorLogic(ScriptedLoadableModuleLogic):

    def hasImageData(self, volumeNode):
        if not volumeNode:
            logging.debug('hasImageData failed: no volume node')
            return False
        if volumeNode.GetImageData() is None:
            logging.debug('hasImageData failed: no image data in volume node')
            return False
        return True

    def convertVTKtoNP(self, vtkVolume):
        imageData = vtkVolume.GetImageData()
        spacing = vtkVolume.GetSpacing()
        r,c,h=imageData.GetDimensions()
        scalars = imageData.GetPointData().GetScalars()
        npMatrix = numpy_support.vtk_to_numpy(scalars)
        npMatrix = npMatrix.reshape(h,c,r)
        return npMatrix, spacing[2::-1]

    def run(self, inputVolume, outputVolume, selectedAxis):

        if not self.hasImageData(inputVolume):
            slicer.util.errorDisplay('Input volume has no image data')
            return False

        logging.info('Processing started')
        serverUrl = 'http://i.kasjen.de/segmentNumpy'
        # input
        inputNode = slicer.util.getNode(inputVolume.GetID())
        imageData, imageSpacing = self.convertVTKtoNP(inputNode)
        logging.info("image is in shape: " + str(imageData.shape))

        # output
        outputNode = slicer.util.getNode(outputVolume.GetID())
        orientedImageData = slicer.modules.segmentations.logic().CreateOrientedImageDataFromVolumeNode(inputNode)
        logging.info("Selected Axis: " + selectedAxis)

        # call server
        logging.info("Calling Server")
        data =  {'imageData':imageData.tolist(), 'imageSpacing': imageSpacing, 'sagitalAxis': 2 if selectedAxis == "X" else 1 if selectedAxis == "Y" else 0}
        maskResponse = requests.post(serverUrl, json=data)
        logging.info("Got Response")
        
        maskNP = np.array(maskResponse.json())
        vtkVolume = numpy_support.numpy_to_vtk(
            num_array=maskNP.ravel(),
            deep=True,
            array_type=vtk.VTK_FLOAT
        )
        orientedImageData.GetPointData().SetScalars(vtkVolume)
        outputNode.AddSegmentFromBinaryLabelmapRepresentation(
            orientedImageData,
            "AI detection",
            [0, 1, 0]
        )
        # outputNode.GetPointData().SetScalars(numpy_support.numpy_to_vtk(imageNP))
        logging.info('Processing completed')

        return True


class DeepSpinalSegmentorTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_DeepSpinalSegmentor1()

    def test_DeepSpinalSegmentor1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")
        #
        # first, get some data
        #
        import SampleData
        SampleData.downloadFromURL(
            nodeNames='FA',
            fileNames='FA.nrrd',
            uris='http://slicer.kitware.com/midas3/download?items=5767'
        )
        self.delayDisplay('Finished with download and loading')

        volumeNode = slicer.util.getNode(pattern="FA")
        logic = DeepSpinalSegmentorLogic()
        self.assertIsNotNone(logic.hasImageData(volumeNode))
        self.delayDisplay('Test passed!')

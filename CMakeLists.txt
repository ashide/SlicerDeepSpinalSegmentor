cmake_minimum_required(VERSION 3.5)

project(SlicerDeepSpinalSegmentor)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/ashide/SlicerDeepSpinalSegmentor")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Shide Adibi (University of Koblenz), Sabine Bauer (University of Koblenz)")
set(EXTENSION_DESCRIPTION "TODO")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/SlicerDeepSpinalSegmentor.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/SlicerDeepSpinalSegmentor/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(DeepSpinalSegmentor)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

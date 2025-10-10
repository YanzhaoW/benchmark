from conan import ConanFile
from conan.tools.cmake import cmake_layout
from conan.tools.layout import basic_layout


class CompressorRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def requirements(self):
        self.requires("gsl-lite/1.0.1")
        self.requires("benchmark/1.9.4")

    def layout(self):
        cmake_layout(self, build_folder="build")
        self.folders.build="build"
        self.folders.generators="build/generators"

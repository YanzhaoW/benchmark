from conan import ConanFile
from conan.tools.cmake import CMakeToolchain


class CompressorRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps"

    def requirements(self):
        self.requires("benchmark/1.9.4")

    # def layout(self):
    #     cmake_layout(self, build_folder="build")
    #     self.folders.build="build"
    #     self.folders.generators="build/generators"

    def generate(self):
        tc = CMakeToolchain(self)
        tc.user_presets_path = ""
        tc.generate()

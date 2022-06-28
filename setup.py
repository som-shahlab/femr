from __future__ import annotations


from typing import Any, Dict

import setuptools
from setuptools.command.build_ext import build_ext
import pathlib

import sysconfig
import os
import subprocess
import shutil
import sys

import numpy as np


class BazelExtension(setuptools.Extension):
    def __init__(self, name: str, target: str, sourcedir: str):
        super().__init__(name, sources=[])
        self.target = target
        self.sourcedir = str(pathlib.Path(sourcedir).resolve())


class cmake_build_ext(build_ext):
    def build_extensions(self) -> None:
        bazel_extensions = [
            a for a in self.extensions if isinstance(a, BazelExtension)
        ]

        if bazel_extensions:
            try:
                bazel_version = subprocess.check_output(
                    ["bazel", "version"]
                ).decode("utf8")
            except OSError:
                raise RuntimeError("Cannot find bazel executable")

        for ext in bazel_extensions:
            python_include_lib = sysconfig.get_config_var("INCLUDEPY")
            if python_include_lib is None:
                raise RuntimeError(
                    "INCLUDEPY did not point to the correct include directory"
                )

            numpy_include_lib = np.get_include()

            python_target_location = os.path.join(ext.sourcedir, "python")
            numpy_target_location = os.path.join(ext.sourcedir, "numpy")
            if os.path.lexists(python_target_location):
                os.remove(python_target_location)

            if os.path.lexists(numpy_target_location):
                os.remove(numpy_target_location)

            os.symlink(python_include_lib, python_target_location)
            os.symlink(numpy_include_lib, numpy_target_location)

            source_env = dict(os.environ)
            env = {
                **source_env,
            }

            subprocess.run(
                args=["bazel", "build", "-c", "dbg", ext.target],
                cwd=ext.sourcedir,
                env=env,
                check=True,
            )

            parent_directory = os.path.abspath(
                os.path.join(self.get_ext_fullpath(ext.name), os.pardir)
            )

            os.makedirs(parent_directory, exist_ok=True)

            shutil.copy(
                os.path.join(ext.sourcedir, "bazel-bin", ext.target),
                self.get_ext_fullpath(ext.name),
            )

            os.chmod(self.get_ext_fullpath(ext.name), 0o700)


print("wat")

setuptools.setup(
    ext_modules=[BazelExtension("piton.extension", "extension.so", "native"),],
    cmdclass={"build_ext": cmake_build_ext},
    zip_safe=False,
)

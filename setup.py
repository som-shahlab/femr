from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import setuptools
from setuptools.command.build_ext import build_ext


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
                subprocess.check_output(["bazel", "version"]).decode("utf8")
            except OSError:
                raise RuntimeError("Cannot find bazel executable")

        for ext in bazel_extensions:
            source_env = dict(os.environ)
            env = {
                **source_env,
            }

            extra_args = []

            if source_env.get("DISTDIR"):
                extra_args.extend(["--distdir", source_env["DISTDIR"]])

            subprocess.run(
                args=["bazel", "build", "-c", "opt", ext.target] + extra_args,
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


setuptools.setup(
    ext_modules=[
        BazelExtension("piton.extension", "extension.so", "native"),
    ],
    cmdclass={"build_ext": cmake_build_ext},
    zip_safe=False,
)

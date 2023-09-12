from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
from typing import List

import setuptools
from setuptools.command.build_ext import build_ext


class BazelExtension(setuptools.Extension):
    def __init__(self, name: str, target: str, sourcedir: str):
        super().__init__(name, sources=[])
        self.target = target
        self.sourcedir = str(pathlib.Path(sourcedir).resolve())


def has_nvcc():
    try:
        subprocess.check_output(["nvcc", "--version"]).decode("utf8")
        return True
    except OSError:
        return False


def can_build_simple(sourcedir, env, bazel_extra_args):
    try:
        subprocess.run(
            args=["bazel"] + bazel_extra_args + ["build", "-c", "opt", "simple_test"],
            cwd=sourcedir,
            env=env,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


class cmake_build_ext(build_ext):
    def build_extensions(self) -> None:
        bazel_extensions = [a for a in self.extensions if isinstance(a, BazelExtension)]

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

            bazel_extra_args: List[str] = []
            extra_args: List[str] = []

            if source_env.get("DISTDIR"):
                extra_args.extend(["--distdir", source_env["DISTDIR"]])

            if has_nvcc():
                extra_args.extend(["--//:cuda=enabled"])

            if source_env.get("MACOSX_DEPLOYMENT_TARGET"):
                extra_args.extend(["--macos_minimum_os", source_env["MACOSX_DEPLOYMENT_TARGET"]])

            if source_env.get("DISABLE_CPU_ARCH") or not can_build_simple(
                sourcedir=ext.sourcedir, env=env, bazel_extra_args=bazel_extra_args
            ):
                bazel_extra_args.extend(["--noworkspace_rc", "--bazelrc=backupbazelrc"])
                assert can_build_simple(
                    sourcedir=ext.sourcedir, env=env, bazel_extra_args=bazel_extra_args
                ), "Cannot build C++ extension"

            subprocess.run(
                args=["bazel", "clean", "--expunge"],
                cwd=ext.sourcedir,
                env=env,
                check=True,
            )

            if source_env.get("DEBUG", False):
                compile_mode = "dbg"
            else:
                compile_mode = "opt"

            subprocess.run(
                args=["bazel"] + bazel_extra_args + ["build", "-c", compile_mode, ext.target] + extra_args,
                cwd=ext.sourcedir,
                env=env,
                check=True,
            )

            parent_directory = os.path.abspath(os.path.join(self.get_ext_fullpath(ext.name), os.pardir))

            os.makedirs(parent_directory, exist_ok=True)

            shutil.copy(
                os.path.join(ext.sourcedir, "bazel-bin", ext.target),
                self.get_ext_fullpath(ext.name),
            )

            os.chmod(self.get_ext_fullpath(ext.name), 0o700)


setuptools.setup(
    ext_modules=[
        BazelExtension("femr.extension", "extension.so", "native"),
    ],
    cmdclass={"build_ext": cmake_build_ext},
    zip_safe=False,
)

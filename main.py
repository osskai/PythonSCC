# -*- coding: utf-8 -*-

import argparse
import subprocess
import pathlib
import sys
from clexer import Lexer
from cparser import Parser
from cutils import *
from csyntax import Syntax
from casmgen import GenASM


def get_arguments():
    """Get the command-line arguments.

    This function sets up the argument parser. Returns a tuple containing
    an object storing the argument values and a list of the file names
    provided on command line.
    """
    desc = """Compile, assemble, and link C files."""
    parser = argparse.ArgumentParser(description=desc, usage="scc [-h] [options] files...")

    # Files to compile
    parser.add_argument("files", metavar="files", nargs="+")

    return parser.parse_args()


def read_file(file):
    """Return the contents of the given file."""
    try:
        source_code = {}
        with open(file) as c_file:
            flat_text = c_file.read()
            c_file.seek(0)
            for count, line in enumerate(c_file):
                source_code[count] = line

            return flat_text, source_code
    except IOError:
        comment = f"could not read file: '{file}'"
        issue_collector.add(ErrorIssue(comment))


def write_asm(asm_source, asm_filename):
    """Save the given assembly source to disk at asm_filename.

    asm_source (str) - Full assembly source code.
    asm_filename (str) - Filename to which to save the generated assembly.

    """
    try:
        with open(asm_filename, "w") as s_file:
            s_file.write(asm_source)
    except IOError:
        comment = f"could not write output file '{asm_filename}'"
        issue_collector.add(ErrorIssue(comment))


def process_file(file, args):
    """Process single file into object file and return the object file name."""
    if file[-2:] == ".c":
        return process_c_file(file, args)
    elif file[-2:] == ".o":
        return file
    else:
        comment = f"unknown file type: '{file}'"
        issue_collector.add(ErrorIssue(comment))


def process_c_file(file, args):
    """Compile a C file into an object file and return the object file name."""
    # 0. source code
    code, source_code = read_file(file)
    if not issue_collector.ok():
        return None

    # 1. lexer
    tokens = Lexer(code)

    # 2. parser
    parser = Parser(tokens)
    tree = parser.parse()
    if not issue_collector.ok():
        return None

    # 3. syntax
    syntax = Syntax(tree)
    syntax.check()
    if not issue_collector.ok():
        return None

    # 4. asm code
    asm_gen = GenASM(tree, source_code)
    asm_source = asm_gen.mask_asm()
    if not issue_collector.ok():
        return None

    asm_file = file[:-2] + ".s"
    obj_file = file[:-2] + ".o"
    write_asm(asm_source, asm_file)
    if not issue_collector.ok():
        return None

    if sys.platform.startswith("darwin"):
        # 5. machine code
        assemble(asm_file, obj_file)
        if not issue_collector.ok():
            return None

    return obj_file


def find_library(file):
    """Search the given library file by searching in common directories.

    If found, returns the path. Otherwise, returns None.
    """
    search_paths = [pathlib.Path("/usr/local/lib/x86_64-linux-gnu"),
                    pathlib.Path("/lib/x86_64-linux-gnu"),
                    pathlib.Path("/usr/lib/x86_64-linux-gnu"),
                    pathlib.Path("/usr/local/lib64"),
                    pathlib.Path("/lib64"),
                    pathlib.Path("/usr/lib64"),
                    pathlib.Path("/usr/local/lib"),
                    pathlib.Path("/lib"),
                    pathlib.Path("/usr/lib"),
                    pathlib.Path("/usr/x86_64-linux-gnu/lib64"),
                    pathlib.Path("/usr/x86_64-linux-gnu/lib")]

    for path in search_paths:
        full_path = path.joinpath(file)
        if full_path.is_file():
            return str(full_path)
    return None


def find_crtnum():
    """Search for the crt0, crt1, or crt2.o files on the system.
    The crt1.o, crti.o, and crtn.o objects comprise the core CRT (C RunTime) objects required to
    enable basic C programs to start and run.
    crt1.o provides the _start symbol that the runtime linker, ld.so.1, jumps to in order to pass control
    to the executable, and is responsible for providing ABI mandated symbols and other process initialization,
    for calling main(), and ultimately, exit(). crti.o and crtn.o provide prologue and epilogue .init and .fini
    sections to encapsulate ELF init and fini code.
    crt1.o is only used when building executables. crti.o and crtn.o are used by executables and shared objects.
    for example:
    cc -c main.c
    ld /usr/lib/crt1.o /usr/lib/crti.o main.o -lc /usr/lib/crtn.o
    ./a.out
    """
    for file in ["crt2.o", "crt1.o", "crt0.o"]:
        crt = find_library(file)
        if crt:
            return crt

    comment = "could not find crt0.o, crt1.o, or crt2.o for linking"
    issue_collector.add(ErrorIssue(comment))
    return None


def find_library_or_err(file):
    """Search the given library file and return path if found.

    If not found, add an error to the error collector and return None.
    """
    path = find_library(file)
    if not path:
        comment = f"could not find {file}"
        issue_collector.add(ErrorIssue(comment))
        return None
    else:
        return path


def assemble(asm_name, obj_name):
    """Assemble the given assembly file into an object file."""
    try:
        subprocess.check_call(["as", "-o", obj_name, asm_name])
    except subprocess.CalledProcessError:
        comment = "assembler returned non-zero status"
        issue_collector.add(ErrorIssue(comment))


def link(binary_name, obj_names):
    """Assemble the given object files into a binary."""
    try:
        crtnum = find_crtnum()
        if not crtnum:
            return

        crti = find_library_or_err("crti.o")
        if not crti:
            return

        linux_so = find_library_or_err("ld-linux-x86-64.so.2")
        if not linux_so:
            return

        crtn = find_library_or_err("crtn.o")
        if not crtn:
            return

        # find files to link
        subprocess.check_call(
            ["ld", "-dynamic-linker", linux_so, crtnum, crti, "-lc"]
            + obj_names + [crtn, "-o", binary_name])

        return True

    except subprocess.CalledProcessError:
        return False


def main():
    """Run the main compiler script."""

    """
    if not sys.platform.startswith("darwin"):
        comment = "only x86_64 MacOs is supported"
        issue_collector.add(ErrorIssue(comment))
        return 1
    """

    objs = []
    """
    arguments = get_arguments()

    for file in arguments.files:
        objs.append(process_file(file, arguments))
    """
    file = "sample.c"

    objs.append(process_file(file, None))

    if sys.platform.startswith("darwin"):
        if any(not obj for obj in objs):
            return 1
        else:
            if not link("out", objs):
                comment = "linker returned non-zero status"
                issue_collector.add(ErrorIssue(comment))
                return 1
            return 0


if __name__ == '__main__':
    flag = main()
    if flag != 0:
        issue_collector.show()


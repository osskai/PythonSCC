# -*- coding: utf-8 -*-

from contextlib import contextmanager
import copy


class IssueCollector:
    """Class that accumulates all errors and warnings encountered."""

    def __init__(self):
        """Initialize the ErrorCollector with no issues to report."""
        self.issues = []

    def add(self, issue):
        """Add the given error or warning (CompilerError) to list of errors."""
        self.issues.append(issue)

    def ok(self):
        """Return True iff there are no errors."""
        return not any(not issue.issue_type == "warning" for issue in self.issues)

    def show(self):  # pragma: no cover
        """Display all warnings and errors."""
        for issue in self.issues:
            print(issue)

    def clear(self):
        """Clear all warnings and errors. Intended only for testing use."""
        self.issues = []


issue_collector = IssueCollector()


# the implementation of exception the ErrorCollector will hold
class BaseIssue(Exception):
    """Base Exception"""

    def __init__(self, comment):
        self.comment = comment
        self.issue_type = None

    def __str__(self):
        """Return a pretty-printable statement of the error."""
        reset_color = "\x1B[0m"

        return f"{reset_color}{self.comment}"


class WarningIssue(BaseIssue):
    """Warning Exception"""

    def __init__(self, comment):
        BaseIssue.__init__(self, comment)
        self.issue_type = "warning"

    def __str__(self):
        bold_color = "\033[1m"
        warn_color = "\x1B[33m"

        warn_str = f"{bold_color}SCC->{warn_color}{self.issue_type}:"
        warn_str += super().__str__()

        return warn_str


class ErrorIssue(BaseIssue):
    """Class representing compile-time errors."""

    def __init__(self, comment):
        BaseIssue.__init__(self, comment)
        self.issue_type = "error"

    def __str__(self):  # pragma: no cover
        """Return a pretty-printable statement of the error."""
        error_color = "\x1B[31m"
        bold_color = "\033[1m"

        err_str = f"{bold_color}SCC->{error_color}{self.issue_type}:"
        err_str += super().__str__()

        return err_str


class ParserError(ErrorIssue):
    """Class representing parser errors."""

    def __init__(self, comment):
        ErrorIssue.__init__(self, comment)


@contextmanager
def parser_error_protect():
    """Wrap this context manager around conditional parsing code.
    it will run the code in [try parsing something]. If an error occurs,
    it will be saved and then [try parsing something else] will run.
    """
    try:
        yield
    except ParserError as e:
        issue_collector.add(e)


class CompilerError(ErrorIssue):
    """Class representing compilation errors."""

    def __init__(self, comment):
        ErrorIssue.__init__(self, comment)


@contextmanager
def compiler_error_protect():
    """Wrap this context manager around conditional parsing code.
        it will run the code in [try parsing something]. If an error occurs,
        it will be saved and then [try parsing something else] will run.
        """
    try:
        yield
    except CompilerError as e:
        issue_collector.add(e)

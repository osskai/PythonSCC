# -*- coding: utf-8 -*-

from ctokens import *
import re
import os
from cutils import *

###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################


class Lexer(object):
    def __init__(self, text):
        # client string input, e.g. "4 + 2 * 3 - 6 / 2"
        self.text = text
        # self.pos is an index into self.text
        self.pos = 0
        self.current_char = self.text[self.pos]
        self.line = 1

    def advance(self):
        """Advance the `pos` pointer and set the `current_char` variable."""
        if self.current_char == '\n':
            self.line += 1

        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]

    def peek(self):
        """only peek, but do not move the position"""
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]

    def handle_error(self, comment):
        issue_collector.add(ErrorIssue(comment))
        return self.get_next_token()

    def skip_whitespace(self):
        # isspace contains '', '\n', '\t', etc.
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def skip_comment(self):
        if self.current_char == '/':
            while self.current_char != '\n' or self.current_char == EOF:
                self.advance()

            self.advance()
        elif self.current_char == '*':
            self.advance()
            while True:
                while self.current_char is not None:
                    if self.current_char == '*' or self.current_char == EOF:
                        break
                    else:
                        self.advance()
                # increase the line
                if self.current_char == '*':
                    self.advance()
                    if self.current_char == '/':
                        self.advance()
                        break
                elif self.current_char is None or self.current_char == EOF:
                    break
                else:
                    comment = f"Unknown Error({self.line})."
                    raise ParserError(comment)

    def number(self):
        """Return a (multidigit) integer or float consumed from the input."""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        token = Token(INT, int(result), lineno=self.line)
        return token

    def identifier(self):
        """Handle identifiers and reserved keywords, and macro definition replace"""
        result = ''
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()

        # if it is a keyword, return it
        # or, a new ID token whose value is the character string (lexeme)
        token = RESERVED_KEYWORDS.get(result.upper(), Token(ID, result))
        token.line = self.line
        return token

    def read_lines(self):
        """Handle all literals"""
        result = ''
        read_next_line = False
        while self.current_char is not None:
            if self.current_char == '/' and (self.peek() == '/' or self.peek() == '*'):
                # skip comments
                self.advance()
                self.skip_comment()
            elif self.current_char == '\\':
                # multiple lines
                read_next_line = True
                self.advance()
                continue
            elif self.current_char == '\n':
                if not read_next_line:
                    break
                else:
                    read_next_line = False
                    self.advance()
                    continue

            result += self.current_char
            self.advance()

        return result

    @staticmethod
    def extract_args(literal):
        literal = literal.lstrip('(')
        literal = literal.rstrip(')')
        literal = literal.replace(' ', '')
        args_pattern = re.compile(r'(?<=,)?(\w+)(?=,)?')
        args_list = re.findall(args_pattern, literal)

        return args_list

    def modules_and_macros(self):
        result = ''
        while self.current_char is not None and self.current_char.isalnum():
            result += self.current_char
            self.advance()

        if result == 'define':
            # only support the basic macro like:
            # #define PI 3.1415
            literals = self.read_lines().strip()
            marco_name_pattern = re.compile(r'\b\w+(?=[(\s])')
            result = re.search(marco_name_pattern, literals)
            if result is None:
                comment = f"Incorrectly Define Macros or Repeatedly Definition({self.line})."
                return self.handle_error(comment)

            # obtain the macro name
            macro_name = result.group(0)
            rest_literals = literals[len(macro_name):]
            if rest_literals[0] == '(':
                defns_pattern = re.compile(r'(?<=[)]).+')
                result = re.search(defns_pattern, rest_literals)
                if result is None:
                    comment = f"Incorrectly Define Macros({self.line})."
                    return self.handle_error(comment)
                # obtain the macro definitions
                defns = result.group(0)
            else:
                # obtain the macro definitions
                defns = rest_literals

            rest_literals = rest_literals[:len(rest_literals)-len(defns)]
            if rest_literals == '':
                # no parameters
                args_list = None
            else:
                args_list = self.extract_args(rest_literals)
                if args_list is None:
                    comment = f"No argument inside parentheses({self.line})."
                    issue_collector.add(WarningIssue(comment))

            original_str = self.text[self.pos:]
            arg_str = macro_name
            if args_list is not None:
                arg_str += '\('
                for i in range(len(args_list)):
                    if i < len(args_list) - 1:
                        arg_str += '\w+,[\s]*'
                    else:
                        arg_str += '\w+'
                arg_str += '\)'

            # match the macro in the text
            macro_pattern = r'\b%s[^\w]' % arg_str
            result = re.findall(macro_pattern, original_str)
            if len(result) == 0:
                macro_pattern = r'\b%s[^\w]' % macro_name
                result = re.search(macro_pattern, original_str)
                if result is not None:
                    comment = f"Incorrectly used macro({self.line})."
                    return self.handle_error(comment)
                else:
                    comment = f"Unused macro {macro_name}({self.line})."
                    issue_collector.add(WarningIssue(comment))
            else:
                for node in result:
                    macro_defns = defns
                    node_str = node[len(macro_name)+1:len(node)-1]
                    parms_list = self.extract_args(node_str)
                    for k in range(len(parms_list)):
                        macro_defs_parm_pattern = re.compile(r'\b%s\b' % args_list[k])
                        macro_defns = re.sub(macro_defs_parm_pattern, '%s' % parms_list[k], macro_defns)

                    replaces_str = ' {}{}'.format(macro_defns, node[-1])
                    result = re.sub(macro_pattern, replaces_str, original_str, 1)
                    # reset the text
                    self.text = result
                    original_str = self.text

                self.pos = 0

            return self.get_next_token()
        elif result == 'include':
            # load modules
            file_name = self.read_lines().strip()
            file_name = file_name.lstrip('<')
            file_name = file_name.rstrip('>')
            file_name = file_name.replace(' ', '')
            # current file path
            file_dir = os.path.dirname(os.path.realpath(__file__))
            file_name = os.path.join(file_dir, file_name)
            try:
                with open(file_name, 'r') as f:
                    new_text = f.read()
                    new_text += self.text[self.pos:]
                    self.text = new_text
                    self.pos = 0
            except IOError:
                comment = f"Cannot find the file(s)({self.line})."
                issue_collector.add(ErrorIssue(comment))

            return self.get_next_token()
        else:
            comment = f"Unsupported macros({self.line})."
            return self.handle_error(comment)

    def string_literal(self, ch):
        result = ''

        while True:
            if self.current_char == ch:
                self.advance()
                break
            elif self.current_char == '\\':
                self.advance()
                if self.current_char == '0':
                    c = '\0'
                elif self.current_char == 't':
                    c = '\t'
                elif self.current_char == 'n':
                    c = '\n'
                elif self.current_char == '\'':
                    c = '\''
                elif self.current_char == '\"':
                    c = '\"'
                elif self.current_char == '\\':
                    c = '\\'
                else:
                    c = self.current_char
                    if '!' <= c <= '~':
                        comment = f"Undesired characters({self.line})."
                        raise ParserError(comment)

                result += c
            else:
                result += self.current_char

            self.advance()

        return Token(CHAR, result)

    def get_next_token(self):
        """Lexical analyzer (also known as scanner or tokenizer)
        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
        while self.current_char is not None:

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isalpha() or self.current_char == '_':
                return self.identifier()

            if self.current_char.isdigit():
                return self.number()

            if self.current_char == '\'' or self.current_char == '\"':
                current_char = self.current_char
                self.advance()
                return self.string_literal(current_char)

            if self.current_char == '#':
                self.advance()
                return self.modules_and_macros()

            if self.current_char == '=':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(REEQ, '==', lineno=self.line)
                else:
                    return Token(ASSIGN, '=', lineno=self.line)

            if self.current_char == '!':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(RENEQ, '!=', lineno=self.line)

            if self.current_char == '<':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(RELEQ, '<=', lineno=self.line)
                else:
                    return Token(RELT, '<', lineno=self.line)

            if self.current_char == '>':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(REGEQ, '>=', lineno=self.line)
                else:
                    return Token(REGT, '>', lineno=self.line)

            if self.current_char == ';':
                self.advance()
                return Token(SEMI, ';', lineno=self.line)

            if self.current_char == ',':
                self.advance()
                return Token(COMMA, ',', lineno=self.line)

            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+', lineno=self.line)

            if self.current_char == '-':
                self.advance()
                if self.current_char == '>':
                    self.advance()
                    return Token(POINTERTO, '->', lineno=self.line)
                else:
                    return Token(MINUS, '-', lineno=self.line)

            if self.current_char == '.':
                self.advance()
                return Token(DOT, '.', lineno=self.line)

            if self.current_char == '*':
                self.advance()
                return Token(MUL, '*', lineno=self.line)

            if self.current_char == '/':
                self.advance()
                if self.current_char == '/' or self.current_char == '*':
                    self.skip_comment()
                    continue
                else:
                    return Token(DIVIDE, '/', lineno=self.line)

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(', lineno=self.line)

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')', lineno=self.line)

            if self.current_char == '[':
                self.advance()
                return Token(LBRACKET, '[', lineno=self.line)

            if self.current_char == ']':
                self.advance()
                return Token(RBRACKET, ']', lineno=self.line)

            if self.current_char == '{':
                self.advance()
                return Token(BEGIN, '{', lineno=self.line)

            if self.current_char == '}':
                self.advance()
                return Token(END, '}', lineno=self.line)

            if self.current_char == '&':
                self.advance()
                if self.current_char == '&':
                    self.advance()
                    return Token(AND_OP, '&&', lineno=self.line)
                else:
                    return Token(AMPERSAND, '&', lineno=self.line)

            if self.current_char == '|' and self.peek() == '|':
                self.advance()
                self.advance()
                return Token(OR_OP, '||', lineno=self.line)

            if self.current_char == ';':
                self.advance()
                return Token(SEMI, ';', lineno=self.line)

            comment = f"Invalid character {self.current_char}({self.line})."
            raise ParserError(comment)

        return Token(EOF, None, lineno=self.line)

    def print_token(self):
        # we get each token every time
        with parser_error_protect():
            token_type = 'START'
            while token_type != EOF:
                token = self.get_next_token()
                token_type = token.type
                print(token)

        if not issue_collector.ok():
            issue_collector.show()

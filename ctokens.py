# -*- coding: utf-8 -*-


import inspect

# Token types
#
# EOF (end-of-file) token is used to indicate that
# there is no more input left for lexical analysis
# operation
PLUS          = 'PLUS'
MINUS         = 'MINUS'
MUL           = 'MUL'
DIVIDE        = 'DIVIDE'
# relation
REEQ          = 'REEQ'
RENEQ         = 'RENEQ'
RELT          = 'RELT'
RELEQ         = 'RELEQ'
REGT          = 'REGT'
REGEQ         = 'REGEQ'
# assign
ASSIGN        = 'ASSIGN'
# scope
LPAREN        = 'LPAREN'
RPAREN        = 'RPAREN'
LBRACKET      = 'LBRACKET'
RBRACKET      = 'RBRACKET'
BEGIN         = 'BEGIN'
END           = 'END'
# symbol
SEMI          = 'SEMI'
COMMA         = 'COMMA'
EOF           = 'EOF'
# reserved keywords
INT           = 'INT'
CHAR          = 'CHAR'
VOID          = 'VOID'
IF            = 'IF'
ELSE          = 'ELSE'
FOR           = 'FOR'
CONTINUE      = 'CONTINUE'
BREAK         = 'BREAK'
RETURN        = 'RETURN'
STRUCT        = 'STRUCT'
# id
ID            = 'ID'
# special symbol
AMPERSAND     = 'AMPERSAND'
DOT           = 'DOT'
POINTERTO     = 'POINTERTO'
AND_OP        = 'AND_OP'
OR_OP         = 'OR_OP'


class Token(object):
    def __init__(self, _type, value, lineno=None):
        self.type = _type
        self.value = value
        self.line = lineno if lineno is not None else -1

    def __str__(self):
        """String representation of the class instance.
        Examples:
            Token(INTEGER, 3)
            Token(PLUS, '+')
            Token(MUL, '*')
        """
        return 'Line{line}: Token({type}, {value})'.format(
            line=self.line,
            type=self.type,
            value=repr(self.value)
        )

    def __repr__(self):
        return self.__str__()


RESERVED_KEYWORDS = {
    'CHAR': Token('CHAR', 'char'),
    'INT': Token('INT', 'int'),
    'VOID': Token('VOID', 'void'),
    'IF': Token('IF', 'if'),
    'ELSE': Token('ELSE', 'else'),
    'FOR': Token('FOR', 'for'),
    'CONTINUE': Token('CONTINUE', 'continue'),
    'BREAK': Token('BREAK', 'break'),
    'RETURN': Token('RETURN', 'return'),
    'STRUCT': Token('STRUCT', 'struct')
}

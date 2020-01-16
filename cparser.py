# -*- coding: utf-8 -*-

from clexer import *
from astnode import *

###############################################################################
#                                                                             #
#  PARSER                                                                     #
#                                                                             #
###############################################################################


def get_calculated(node):
    result = node.calculate()
    if result is not None:
        result = int(result)
        return Const(result, BaseType('int'))
    else:
        return node


# main function to have the codes parsed
class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        # set current token to the first token taken from the input
        self.current_token = self.lexer.get_next_token()
        self.internal_type = []  # internal variable type
        # for utility
        self.previous_line = []

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if not self.current_token.type == token_type:
            comment = f"token '{token_type}' is expected rather than " \
                      f"'{self.current_token.type}'({self.current_token.line})."

            raise ParserError(comment)

        self.current_token = self.lexer.get_next_token()

    def load_internal_type(self):
        """4 internal types:"""
        self.internal_type.append(INT)
        self.internal_type.append(CHAR)
        self.internal_type.append(VOID)
        self.internal_type.append(STRUCT)

    # main loop to parse the program
    def program(self):
        """"""
        node = TranslationUnit()

        # add base type into external_type
        self.load_internal_type()

        # external declaration
        # some basic problem
        while self.current_token.type != EOF:
            with parser_error_protect():
                decl_node = self.external_declaration()
                for child_node in decl_node.nodes:
                    node.add(child_node)

                continue

            # if there are errors, break current parser process
            break

        # if there are some tokens are not parsed
        if self.current_token.type != EOF:
            comment = f"unexpected token '{self.current_token.value}'({self.current_token.line})"
            raise ParserError(comment)

        return node

    def external_declaration(self):
        """external_declaration : function_definition
                                | declaration
        """
        return self.declaration()

    def declaration(self):
        """declarations : type_specifier (declarator COMMA)* SEMI
        because function_definition has the same structure in the beginning
        for_example:
        int a;
        int a, c;
        int b = 0, c;
        int add(int x, int y);
        or
        int add(int x, int y) { int z; ... }
        """
        # type specifier
        nodes = DeclarationList()
        type_node = self.type_specifier()
        for decl_node in self.init_declarator_list().nodes:
            decl_node.set_type(type_node)
            if decl_node.is_function_definition():
                # function definition
                body_node = self.compound_statement()
                func_node = FunctionDefn(decl_node, body_node,
                                         lineno=decl_node.line, end_lineno=body_node.end_line)
                nodes.add(func_node)
                return nodes

            nodes.add(decl_node)

        self.eat(SEMI)
        return nodes

    def init_declarator_list(self):
        """init_declarator_list : init_declarator
                                | init_declarator_list, init_declarator
        """
        nodes = VariableList()
        # it may be a struct declaration
        if self.current_token.type == SEMI:
            nodes.add(Declaration(lineno=self.current_token.line))
            return nodes

        while self.current_token.type != SEMI:
            child_node = self.init_declarator()
            nodes.add(child_node)
            if self.current_token.type == SEMI:
                break
            elif self.current_token.type == BEGIN:
                """function_definition : type_specifier declarator compound_statement"""
                # function definition
                child_node.function_definition = True
                break

            self.eat(COMMA)

        return nodes

    def init_declarator(self):
        """init_declarator : declarator
                           | declarator ASSIGN initializer
        """
        node = self.declarator()
        if self.current_token.type == ASSIGN:
            self.eat(ASSIGN)
            node.add_init_value(self.initializer())

        return node

    def initializer(self):
        """initializer : assignment_expression
                       | BEGIN initializer_list END
        for example:
        char str[3] = {'a', 'b', 'c'};
        char strr[2][3] = {{}, {}}
        int arr[3] = {1, 2, };
        int a = b
        """
        if self.current_token.type == BEGIN:
            self.eat(BEGIN)
            node = self.initializer_list()
            self.eat(END)
            return node
        else:
            return self.assignment_expression()

    def initializer_list(self):
        """initializer_list : initializer
                            | initializer_list, initializer
        """
        nodes = InitializerList()
        while self.current_token.type != END:
            nodes.add(self.initializer())
            if self.current_token.type == END:
                break

            self.eat(COMMA)

        return nodes

    def type_specifier(self):
        """type_spec : INT
                     | CHAR
                     | VOID
                     | struct_specifier
        for example:
        int/char/void/struct/variable
        return: a Type
        """
        token = self.current_token
        if token.type in (INT, CHAR, VOID):
            self.eat(token.type)
            return BaseType(token.value)
        elif token.type == STRUCT:
            return self.struct_specifier()
        else:
            comment = f"undefined type '{token.value}'({token.line})."
            raise ParserError(comment)

    def struct_specifier(self):
        """struct_spec : struct ID (BEGIN struct_declaration_list END)
        for example:
        struct node {
            int a, b;
        }
        struct node s;
        """
        self.eat(STRUCT)
        struct_name = self.variable()
        expr_list = EmptyNode()
        if self.current_token.type == BEGIN:
            self.eat(BEGIN)
            expr_list = self.struct_delaration_list()
            self.eat(END)

        return StructType(struct_name, expr_list)

    def struct_delaration_list(self):
        """struct_delaration_list : struct_declaration
                                  | struct_delaration_list struct_delaration
        """
        node = StructMembersList()
        while self.current_token.type != END:
            for child_node in self.struct_declaration().nodes:
                node.add(child_node)

        return node

    def struct_declaration(self):
        """struct_delaration : type_specifier struct_declarator_list SEMI
        for example:
        int a, b;
        """
        nodes = DeclarationList()
        type_node = self.type_specifier()
        for child_node in self.struct_declarator_list().nodes:
            child_node.set_type(type_node)
            nodes.add(child_node)

        self.eat(SEMI)
        return nodes

    def struct_declarator_list(self):
        """struct_declarator_list : declarator
                                  | struct_declarator_list, declaraor
        for example:
        a, b;
        """
        nodes = NodeList()
        while self.current_token.type != SEMI:
            child_node = self.declarator()
            nodes.add(child_node)
            if self.current_token.type == SEMI:
                break

            self.eat(COMMA)

        return nodes

    def declarator(self):
        """declarator : pointer direct_declarator
                      | direct_declarator
        for example:
        a
        *a
        """
        if self.current_token.type == MUL:
            self.eat(MUL)
            node = self.declarator()
            # this type in the outer_most
            node.set_type(PointerType())
            return node

        node = self.direct_declarator()

        return node

    def direct_declarator(self):
        """direct_declarator : ID
                             | direct_declarator LBRACKET const_expr RBRACKET
                             | direct_declarator LBRACKET RBRACKET
                             | direct_declarator LPAREN RPAREN
                             | direct_declarator LPAREN parameter_type_list RPAREN
        for example:
        add()
        add(int a, int b)
        str[2]
        not support function pointer 00087.c, 00088.c, 00089.c
        """
        token = self.current_token
        node = Declaration(self.variable(), lineno=token.line)
        if self.current_token.type == LPAREN:
            # function type
            self.eat(LPAREN)
            node.add_type(FunctionType(self.parameter_type_list()))
            self.eat(RPAREN)
        elif self.current_token.type == LBRACKET:
            # array type, only support const int type,
            # and for const int a = 100, a is also supported
            self.eat(LBRACKET)
            node.add_type(ArrayType(self.const_expression()))
            self.eat(RBRACKET)

        return node

    def variable(self):
        """variable : ID"""
        name = self.current_token.value
        self.eat(ID)
        return name

    def parameter_type_list(self):
        """parameter_type_list : parameter_declaration
                              | parameter_type_list COMMA parameter_declaration
        for example:
        int a, int b
        """
        nodes = ParameterList()
        while self.current_token.type != RPAREN:
            child_node = self.parameter_declaration()
            nodes.add(child_node)
            if self.current_token.type == RPAREN:
                break

            self.eat(COMMA)

        return nodes

    def parameter_declaration(self):
        """parameter_declaration : type_specifier declarator
        for example:
        int i
        """
        # must follow this format
        type_node = self.type_specifier()
        decl_node = self.declarator()

        decl_node.set_type(type_node)

        return decl_node

    def compound_statement(self):
        """
        compound_statement: BEGIN (declaration)* (statement)* END
        compound_statement is a main body, for example:
        int main()
        {           -->
            ...     --> compound_statement -> Block
        }           -->
        """
        token = self.current_token
        self.previous_line.append(token.line)
        self.eat(BEGIN)

        declaration_nodes = DeclarationList()
        statement_nodes = StatementsList()
        while True:
            with parser_error_protect():
                if self.current_token.type in self.internal_type:
                    decl_node = self.declaration()
                    for child in decl_node.nodes:
                        declaration_nodes.add(child)

                    continue

            if self.current_token.type == END:
                break

            with parser_error_protect():
                stmt_node = self.statement()
                statement_nodes.add(stmt_node)
                continue

            break

        end_token = self.current_token
        self.eat(END)
        if len(self.previous_line) == 0:
            self.previous_line.append(token.line)

        node = CompoundStatement(declaration_nodes, statement_nodes,
                                 lineno=self.previous_line.pop(), end_lineno=end_token.line)

        return node

    def statement(self):
        """
        statement : compound_statement
                  | expr_statement
                  | selection_statement
                  | iteration_statement
                  | jump_statement
        """
        if self.current_token.type == BEGIN:
            node = self.compound_statement()
        elif self.current_token.type == IF:
            node = self.selection_statement()
        elif self.current_token.type in (RETURN, BREAK, CONTINUE):
            node = self.jump_statement()
        elif self.current_token.type == FOR:
            node = self.iteration_statement()
        elif self.current_token.type == SEMI:
            self.eat(SEMI)
            node = EmptyNode()
        else:
            node = self.expression_statement()
        return node

    def selection_statement(self):
        """
        if_statement : IF LPAREN expr RPAREN statement { ELSE statement }
        for example:
        if (a > b)
            int c = 1;
        else
            int d = 0;
        """
        token = self.current_token
        self.eat(IF)
        self.eat(LPAREN)
        expr_node = self.expression()
        self.eat(RPAREN)

        self.previous_line.append(token.line)
        then_node = self.statement()

        else_node = EmptyNode()
        if self.current_token.type == ELSE:
            self.eat(ELSE)
            self.previous_line.append(token.line)
            else_node = self.statement()

        return IfStatement(expr_node, then_node, else_node, lineno=self.previous_line.pop())

    def iteration_statement(self):
        """
        for_statement : FOR LPAREN expr_statement expr_statement expr RPAREN statement
        for example:
        for (int a = 0; a < 10; a = a + 1)
        {
            ...
        }
        """
        token = self.current_token
        self.eat(FOR)
        self.eat(LPAREN)
        # contain semi
        if self.current_token.type in self.internal_type:
            begin_stmt = self.declaration()
        else:
            begin_stmt = self.expression_statement()
        # terminal condition
        end_stmt = self.expression_statement()
        # add or minus condition
        expr_node = self.expression()
        self.eat(RPAREN)

        # store the current token
        self.previous_line.append(token.line)
        stmt_node = self.statement()

        return ForLoop(begin_stmt, expr_node, end_stmt, stmt_node, lineno=self.previous_line.pop())

    def jump_statement(self):
        """
        jump_statement : CONTINUE SEMI
                       | BREAK SEMI
                       | RETURN expr
        """
        token = self.current_token
        if token.type == CONTINUE:
            self.eat(CONTINUE)
            node = ContinueStatement(lineno=token.line)
        elif token.type == BREAK:
            self.eat(BREAK)
            node = BreakStatement(lineno=token.line)
        elif token.type == RETURN:
            self.eat(RETURN)
            if self.current_token.type == SEMI:
                node = ReturnStatement(EmptyNode(), token.line)
            else:
                node = ReturnStatement(self.expression(), token.line)
        else:
            node = EmptyNode()

        self.eat(SEMI)

        return node

    def expression_statement(self):
        """expression_statement: expression SEMI
        for example:
        a = 1;
        """
        # empty expression statement
        with parser_error_protect():
            if self.current_token.type in self.internal_type:
                token = self.current_token

                declaration_nodes = DeclarationList()
                decl_node = self.declaration()
                for child in decl_node.nodes:
                    declaration_nodes.add(child)
            else:
                stmt_node = ExpressionStatement(self.expression(), lineno=self.current_token.line)
                self.eat(SEMI)
                return stmt_node

        return CompoundStatement(declaration_nodes, StatementsList(), lineno=token.line)

    def expression(self):
        """
        expression : assignment_expression
        for example:
        a = 1
        """
        return self.assignment_expression()

    def const_expression(self):
        """const_expression : logical_or_expression
        """
        return self.logical_or_expression()

    def assignment_expression(self):
        """assignment_expression : logical_or_expression
                                 | unary_expression ASSIGN assignment_expression
        """
        node = self.logical_or_expression()
        token = self.current_token
        if token.type == ASSIGN:
            # only support for unary_expression
            if isinstance(node, BinOp):
                comment = f"Not Support in current operation {node.op}({token.line})"
                issue_collector.add(ErrorIssue(comment))

            self.eat(ASSIGN)
            right_node = self.assignment_expression()
            node = BinOp(left=node, op=token.value, right=right_node, lineno=token.line)

        return node

    def logical_or_expression(self):
        """logical_or_expression : logical_add_expression
                                 | logical_or_expression OR_OP logical_add_expression
        for example:
        a || b
        """
        node = self.logical_add_expression()

        if self.current_token.type == OR_OP:
            token = self.current_token
            self.eat(token.type)

            node = BinOp(left=node, op=token.value, right=self.logical_add_expression(), lineno=token.line)

        return node

    def logical_add_expression(self):
        """logical_add_expression : equality_expression
                                  | logical_add_expression AND_OP equality_expression
        for example:
        a && b
        """
        node = self.equality_expression()

        if self.current_token.type == AND_OP:
            token = self.current_token
            self.eat(token.type)

            node = BinOp(left=node, op=token.value, right=self.equality_expression(), lineno=token.line)

        return node

    def equality_expression(self):
        """
        equality_expression : relation_expression
                            | equality_expression (EQ | NEQ) relation_expression
        for example:
        a == b
        a != b
        """
        node = self.relation_expression()

        if self.current_token.type in (REEQ, RENEQ):
            token = self.current_token
            self.eat(token.type)

            node = get_calculated(BinOp(left=node, op=token.value, right=self.relation_expression(), lineno=token.line))

        return node

    def relation_expression(self):
        """
        relation_expression : additive_expression
                            | relation_expression (LT | GT | LEQ | GEQ) additive_expression
        for example:
        a >= b
        """
        node = self.additive_expression()

        while self.current_token.type in (RELT, REGT, RELEQ, REGEQ):
            token = self.current_token
            self.eat(token.type)

            node = get_calculated(BinOp(left=node, op=token.value, right=self.additive_expression(), lineno=token.line))

        return node

    def additive_expression(self):
        """
        additive_expr : multiple_expression
                      | additive_expression (PLUS | MINUS) multiple_expression
        for example:
        a + b
        a + b - c
        """
        node = self.multiple_expression()

        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            self.eat(token.type)

            node = get_calculated(BinOp(left=node, op=token.value, right=self.multiple_expression(), lineno=token.line))

        return node

    def multiple_expression(self):
        """term : unary_expression
                | multiple_expression (MUL | DIV) unary_expression
        for example:
        a * b / c
        """
        node = self.unary_expression()

        while self.current_token.type in (MUL, DIVIDE):
            token = self.current_token
            self.eat(token.type)

            node = get_calculated(BinOp(left=node, op=token.value, right=self.unary_expression(), lineno=token.line))

        return node

    def unary_expression(self):
        """unary_expression : postfix_expression
                            | unary_operator unary_expression
        for example:
        +a
        return: UnaryOp or a Const
        """
        token = self.current_token
        if token.type == PLUS:
            self.eat(PLUS)
            node = self.unary_expression()
        elif token.type == MINUS:
            self.eat(MINUS)
            node = get_calculated(Negative(self.unary_expression(), lineno=token.line))
        elif token.type == MUL:
            self.eat(MUL)
            node = Pointer(self.unary_expression(), lineno=token.line)
        elif token.type == AMPERSAND:
            self.eat(AMPERSAND)
            node = AddrOf(self.unary_expression(), lineno=token.line)
        else:
            node = self.postfix_expression()

        return node

    def postfix_expression(self):
        """
        postfix_expr : primary_expression
                     | primary_expression LPAREN argument_expression_list RPAREN
                     | primary_expression LPAREN RPAREN
                     | primary_expression LBRACKET expression RBRACKET
                     | primary_expression DOT ID
                     | primary_expression PTR_OP ID
        for example:
        fun(a, b, c)
        b[a+2]
        a.b
        a->b
        """
        node = self.primary_expression()

        token = self.current_token
        if token.type == LPAREN:
            # function expr cannot link more postfix expressions
            self.eat(LPAREN)
            arg_node = self.argument_expression_list()
            node = FunctionOp(node, arg_node, lineno=token.line)
            self.eat(RPAREN)

        if self.current_token.type == LBRACKET:
            self.eat(LBRACKET)
            expr_node = self.expression()
            node = ArrayOp(node, expr_node, lineno=token.line)
            self.eat(RBRACKET)

        if self.current_token.type in (DOT, POINTERTO):
            token = self.current_token
            self.eat(token.type)
            node = StructOp(left=node, op=token.value, right=self.postfix_expression(), lineno=token.line)

        return node

    def primary_expression(self):
        """
        primary_expr : CONSTANT
                     | LPAREN expr RPAREN
                     | variable
        constant variables
        3, 1.0, 'Hello World!', add, (a=b), empty
        """
        token = self.current_token
        if token.type == INT:
            self.eat(INT)
            return Const(int(token.value), BaseType('int'))
        elif token.type == CHAR:
            self.eat(CHAR)
            return StringLiteral(token.value)  # using eval has problem
        elif token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expression()
            self.eat(RPAREN)
            return node
        elif token.type == ID:
            node = Id(token.value, lineno=token.line)
            self.eat(ID)
            return node
        else:
            # cannot enter into this scope
            return EmptyNode()

    def argument_expression_list(self):
        """
        argument_expression_list : assignment_expression
                                 | argument_expression_list COMMA assignment_expression
        for example:
        a, b, c
        """
        nodes = ParameterList()
        while self.current_token.type != RPAREN:
            child_node = self.assignment_expression()
            nodes.add(child_node)
            if self.current_token.type == RPAREN:
                break

            self.eat(COMMA)

        return nodes

    def parse(self):
        with parser_error_protect():
            node = self.program()
            return node

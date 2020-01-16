# -*- coding: utf-8 -*-


###############################################################################
#                                                                             #
#  AST tree                                                                    #
#                                                                             #
###############################################################################


class AST(object):
    """the base AST node"""
    def is_null(self):
        return False

    def is_const(self):
        return False

    def is_lvalue(self):
        return self.__dict__.get('lvalue')

    def set_lvalue(self):
        self.lvalue = True

    def is_oaddr(self):
        return self.__dict__.get('oaddr')

    def set_oaddr(self):
        self.oaddr = True

    def calculate(self):
        """if a node corresponds to the expression "5+3", then this method would return 8."""
        return None

    def visit(self, visitor):
        """referred by visitor"""
        return self.dig_visit(self.__class__, visitor)

    def dig_visit(self, node, visitor):
        """appending the class' name with 'visit_'"""
        method_name = 'visit' + self.camel_to_lower_case(node.__name__)
        visitor_method = getattr(visitor, method_name, None)
        if visitor_method is None:
            # call the class's superclass
            bases = node.__bases__
            last = None
            for child in bases:
                last = self.dig_visit(child, visitor)
            return last
        else:
            return visitor_method(self)

    @staticmethod
    def camel_to_lower_case(name):
        # base class remains the special case
        if name == 'AST':
            return '_ast'

        lower_name = []
        for item in name:
            if item.isupper():
                lower_name.append('_')
            lower_name.append(item)

        return "".join(lower_name).lower()


class EmptyNode(AST):
    """empty node"""
    def __init__(self):
        self.type = 'void'

    def is_null(self):
        return True


class StringLiteral(AST):
    """A string literal, e.g. the string "Hello World" in
    printf("Hello World")."""
    def __init__(self, _str):
        self.expr = _str
        self.type = PointerType(BaseType('char'))

    def get_str(self):
        return self.expr


class Const(AST):
    """ A const value, like '5', 'hello world'"""
    def __init__(self, value, _type):
        self.expr = value
        self.type = _type

    def calculate(self):
        return self.expr

    def is_const(self):
        return True


class UnaryOp(AST):
    """any generic unary operator"""
    def __init__(self, node, lineno=None):
        self.expr = node
        self.line = lineno if lineno is not None else -1


class Id(UnaryOp):
    """an identifier, such as function name or variable"""
    pass


class Pointer(UnaryOp):
    """a pointer refer, e.g. '*a'"""
    pass


class AddrOf(UnaryOp):
    """an address-of operator, e.g. '&a'"""
    pass


class Negative(UnaryOp):
    """A negative unary operator, e.g. '-5'."""
    def calculate(self):
        value = self.expr.calculate()
        if value is not None:
            return -value

        return None


class ArrayOp(UnaryOp):
    """array operator, such as 'a[5+b]'"""
    def __init__(self, expr, index, lineno=None):
        UnaryOp.__init__(self, expr, lineno)
        self.index = index


class FunctionOp(UnaryOp):
    """function operator, such as func(a, b, c)"""
    def __init__(self, name, args, lineno=None):
        UnaryOp.__init__(self, name, lineno)
        self.args = args


class StructOp(UnaryOp):
    """any binary operator for struct, ., ->"""
    TO_OPS = ['.', '->']

    def __init__(self, left, op, right, lineno=None):
        UnaryOp.__init__(self, right, lineno)
        self.parent = left
        self.op = op


class BinOp(AST):
    """any binary operator, (+/-/*//), (=)"""
    def __init__(self, left, op, right, lineno=None):
        self.left = left
        self.op = op
        self.right = right
        self.line = lineno if lineno is not None else -1

    def calculate(self):
        left_value = self.left.calculate()
        right_value = self.right.calculate()
        if left_value is not None and right_value is not None:
            return int(eval(f"{left_value} {self.op} {right_value}"))
        else:
            return None


class StatementAST(AST):
    """statement base class"""
    def __init__(self, lineno=None):
        self.is_needed = True
        self.has_return = False
        self.line = lineno if lineno is not None else -1

    def set_to_ignore(self):
        self.is_needed = False


class IfStatement(StatementAST):
    """if else statement"""
    def __init__(self, expr, then_stmt, else_stmt, lineno=None):
        StatementAST.__init__(self, lineno)
        self.expr = expr
        self.then_stmt = then_stmt
        self.else_stmt = else_stmt


class BreakStatement(StatementAST):
    """A break statement."""
    pass


class ContinueStatement(StatementAST):
    """A continue statement."""
    pass


class ReturnStatement(StatementAST):
    """return statement"""
    def __init__(self, expr, lineno=None):
        StatementAST.__init__(self, lineno)
        self.expr = expr
        self.is_final_one = False

    def is_final(self):
        return self.is_final_one


class ForLoop(StatementAST):
    """for loop"""
    def __init__(self, begin_stmt, expr, end_stmt, stmt, lineno=None):
        StatementAST.__init__(self, lineno)
        self.begin_stmt = begin_stmt
        self.end_stmt = end_stmt
        self.expr = expr
        self.stmt = stmt


class ExpressionStatement(StatementAST):
    """A expression statement"""
    def __init__(self, expr, lineno=None):
        StatementAST.__init__(self, lineno)
        self.expr = expr

    def is_null(self):
        return self.expr.is_null()


class CompoundStatement(StatementAST):
    """A compound statement, e.g. '{ int i; i += 1; }'."""
    def __init__(self, declaration_list, statement_list, lineno=None, end_lineno=None):
        StatementAST.__init__(self, lineno)
        self.declarations = declaration_list
        self.statements = statement_list
        self.end_line = end_lineno if end_lineno is not None else lineno


# every type, we have a type, and child type, and recursively, child has child type
# so we can define int** PointerType(PointerType(BaseType)), from right to left
#                             ^          ^          ^
#                             |          |          |
#                            type      child      child
class Type(AST):
    """assign a type node to variable"""
    def __init__(self, child=None):
        self.child = child if child is not None else EmptyNode()

    def set_base_type(self, _type):
        if self.child.is_null():
            self.child = _type
        else:
            self.child.set_base_type(_type)

    def get_string(self):
        pass

    def get_outer_string(self):
        pass

    def get_base_type(self):
        return self


class BaseType(Type):
    """A base type representing ints, chars, etc..."""
    def __init__(self, type_str, child=None):
        Type.__init__(self, child)
        self.type_str = type_str

    def get_string(self):
        return self.type_str

    def get_outer_string(self):
        return self.type_str


class PointerType(Type):
    """A type representing a pointer to another (nested) type."""
    def __init__(self, child=None):
        Type.__init__(self, child)

    def get_string(self):
        return f"pointer({self.child.get_string()})"

    def get_outer_string(self):
        return 'pointer'

    def get_base_type(self):
        if not self.child.is_null():
            return self.child.get_base_type()
        return self


class ArrayType(Type):
    """A type representing an array, e.g. a[], a[5]"""
    def __init__(self, index, child=None):
        Type.__init__(self, child)
        self.index = index

    def get_string(self):
        return 'pointer'

    def get_outer_string(self):
        return 'pointer'

    def get_base_type(self):
        if not self.child.is_null():
            return self.child.get_base_type()
        return self


class StructType(Type):
    """A type represent a struct"""
    def __init__(self, struct_name, expr_list=None, child=None):
        Type.__init__(self, child)
        self.name = struct_name
        self.exprs = expr_list

    def get_string(self):
        return 'struct'

    def get_outer_string(self):
        return 'struct'


class FunctionType(Type):
    """A type representing a function"""
    def __init__(self, parms=None, child=None):
        Type.__init__(self, child)
        self.parms = parms if parms is not None else EmptyNode()

    def get_return_type(self):
        """Returns the return type of the function."""
        return self.child

    def get_string(self):
        parms_str = ''
        for parm in self.parms.nodes:
            parms_str += ',' + parm.type.get_string()

        return 'function(%s)->%s' % (parms_str[1:], self.child.get_string())

    def get_outer_string(self):
        return 'function'


class Declaration(AST):
    """A node representing a declaration of a function or variable."""
    def __init__(self, name=None, _type=None, lineno=None):
        self.type = _type if _type is not None else EmptyNode()
        self.name = name
        self.init_value = None
        self.function_definition = False
        self.is_used = False
        self.line = lineno if lineno is not None else -1

    def set_type(self, _type):
        if self.type.is_null():
            self.type = _type
        else:
            self.type.set_base_type(_type)

    def add_type(self, _type):
        _type.set_base_type(self.type)
        self.type = _type

    def get_base_type(self):
        if not self.type.is_null():
            return self.type.get_base_type()
        else:
            return None

    def add_init_value(self, value):
        self.init_value = value

    def is_function_definition(self):
        return self.function_definition

    def has_initialized(self):
        return self.init_value is not None


class FunctionDefn(AST):
    """A node representing a function definition"""
    def __init__(self, declaration, body, lineno=None, end_lineno=None):
        self.name = declaration.name
        self.type = declaration.type
        self.return_type = declaration.type.child
        self.body = body
        self.has_ignore_parts = False
        self.line = lineno if lineno is not None else -1
        self.end_line = end_lineno if end_lineno is not None else -1

    def get_base_type(self):
        return self.type


class NodeList(AST):
    """A list of nodes"""
    def __init__(self, node=None):
        self.nodes = []
        if node is not None:
            self.nodes.append(node)

    def add(self, node):
        self.nodes.append(node)

    def is_null(self):
        if len(self.nodes) == 0:
            return True

        return False


class TranslationUnit(NodeList):
    """An unit list, derived for identification"""
    pass


class StructMembersList(NodeList):
    """A struct member list, derived for identification"""
    pass


class DeclarationList(NodeList):
    """A declaration list, derived for identification"""
    pass


class InitializerList(NodeList):
    """An initializer list, derived for identification"""
    pass


class VariableList(NodeList):
    """A variable list, derived for identification"""
    pass


class ParameterList(NodeList):
    """A parameter list, derived for identification"""
    pass


class StatementsList(NodeList):
    """A statement list, derived for identification"""
    pass


###############################################################################
#                                                                             #
#  AST visitors (walkers)                                                     #
#                                                                             #
###############################################################################


class NodeVisitor(object):
    def visit(self, node):
        return node.visit(self)

    def visit_list(self, _list):
        """like NodeList in parser"""
        last = None
        for child in _list:
            last = child.visit(self)
        return last



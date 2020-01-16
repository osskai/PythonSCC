# -*- coding: utf-8 -*-

from cparser import *
from collections import OrderedDict
from collections import namedtuple
import sys

###############################################################################
#                                                                             #
#  TABLES, SEMANTIC ANALYSIS                                                  #
#                                                                             #
###############################################################################


# scope type for asm_code generation
STRUCTTYPE = 1
COMMONTYPE = 2


class ScopedSymbolTable(object):
    """scoped symbol table for look up"""
    # Tables.vars
    # Tables.structs
    Tables = namedtuple('Tables', ['vars', 'structs'])

    def __init__(self, name, scope_type=None, enclosing_scope=None):
        self.name = name
        self.type = scope_type if scope_type is not None else COMMONTYPE
        self._symbols = self.Tables(OrderedDict(), OrderedDict())
        # parent scope, append itself
        self.enclosing_scope = enclosing_scope
        if enclosing_scope is not None:
            self.enclosing_scope.children.append(self)
        # the child scopes of current scope
        self.children = []

    @property
    def symbols(self):
        return self._symbols

    # insert symbols
    def insert(self, name, value):
        """add symbol with the given value"""
        if name in self._symbols.vars:
            if not self._symbols.vars[name].type.get_string() == value.type.get_string():
                comment = f"Variable '{name}' as '{self._symbols.vars[name].type.get_string()}' " \
                          f"has declared before({value.line})."
                raise CompilerError(comment)
            else:
                comment = f"Variable '{name}' has declared in line" \
                          f"{self._symbols.vars[name].line} before({value.line})."
                raise CompilerError(comment)

        self._symbols.vars[name] = value

    def insert_struct(self, name, value):
        """add symbol with the given value"""
        if name not in self._symbols.structs:

            self._symbols.structs[name] = value
        else:
            symbol = self.lookup_struct(name)
            if symbol.exprs.is_null():
                # replace the null member one
                self._symbols.structs[name] = value
            else:
                comment = f"Redeclare the struct"
                issue_collector.add(WarningIssue(comment))

    def lookup(self, name, recursively_search=True):
        # 'symbol' is either an instance of the Symbol class or None
        symbol = self._symbols.vars.get(name)

        if symbol is not None:
            return symbol

        # if we do not want search recursively, like searching member inside struct scope
        if not recursively_search:
            return None

        # recursively go up the chain and lookup the name
        if self.enclosing_scope is not None:
            return self.enclosing_scope.lookup(name)
        else:
            return None

    def lookup_struct(self, name):
        symbol = self._symbols.structs.get(name)

        # for the embedded struct declaration, find the upper one
        if symbol is not None and not symbol.exprs.is_null():
            return symbol

        # recursively go up the chain and lookup the name
        if self.enclosing_scope is not None:
            return self.enclosing_scope.lookup_struct(name)
        else:
            return None


class SymbolTableVisitor(NodeVisitor):
    def __init__(self):
        self.current_scope = None
        self.enclosing_scope = None
        self.recursively_search = True

    def push_table(self, node, name, scope_type=None):
        """Pushes a new symbol table onto the visitor's symbol table stack"""
        self.current_scope = ScopedSymbolTable(
            name=name,
            scope_type=scope_type,
            enclosing_scope=self.current_scope  # None
        )
        # every node has its own symbol table, easy for code generation
        node.scope_symbol = self.current_scope

    def pop_table(self):
        # return to parent scope
        self.current_scope = self.current_scope.enclosing_scope

    def visit_ast(self, node):
        # noting to do
        pass

    def visit_id(self, node):
        symbol = self.current_scope.lookup(node.expr, self.recursively_search)
        if symbol is not None:
            # used in the type check
            # obtain the type of the Id node
            node.symbol = symbol
            node.symbol.is_used = True
            if not isinstance(node.symbol.type, FunctionType):
                node.set_lvalue()

            # in StructOp to search the member inside the struct scope
            base_type = node.symbol.get_base_type()
            if isinstance(base_type, StructType):
                if base_type.exprs.is_null():
                    struct_symbol = self.current_scope.lookup_struct(base_type.name)
                    if struct_symbol is not None:
                        base_type.exprs = struct_symbol.exprs
                        node.symbol.scope_symbol = struct_symbol.scope_symbol
                    else:
                        comment = f"Struct '{base_type.name}' not found({node.line})"
                        raise CompilerError(comment)
        else:
            comment = f"Identifier '{node.expr}' not found({node.line})"
            raise CompilerError(comment)

    def visit_unary_op(self, node):
        """Pointer, AddrOf, and Negative"""
        node.expr.visit(self)
        node.symbol = node.expr.symbol

    def visit_function_op(self, node):
        """FunctionOp"""
        node.expr.visit(self)
        node.args.visit(self)

    def visit_array_op(self, node):
        node.expr.visit(self)
        node.symbol = node.expr.symbol
        # if enter a new area, we need change the current scope
        super_recursive = self.recursively_search
        self.recursively_search = True
        self.current_scope = self.enclosing_scope
        node.index.visit(self)
        self.recursively_search = super_recursive

    def visit_struct_op(self, node):
        node.parent.visit(self)
        # it is special that we need to check only in the struct itself
        self.recursively_search = False
        self.current_scope = node.parent.symbol.scope_symbol
        node.expr.visit(self)
        node.symbol = node.expr.symbol
        self.recursively_search = True
        self.current_scope = self.enclosing_scope

    def visit_bin_op(self, node):
        node.left.visit(self)
        node.right.visit(self)

    def visit_node_list(self, node):
        self.visit_list(node.nodes)

    def visit_translation_unit(self, node):
        """the entry of the file"""
        self.current_scope = ScopedSymbolTable(
            name="entry",
            enclosing_scope=self.current_scope  # None
        )
        self.visit_node_list(node)
        node.scope_symbol = self.current_scope

    def add_symbol(self, node):
        # for struct Point* s; the type is PointerType(StructType)
        base_type = node.get_base_type()
        if not node.name:
            self.current_scope.insert_struct(base_type.name, base_type)
        else:
            self.current_scope.insert(node.name, node)
            if isinstance(base_type, StructType):
                self.current_scope.insert_struct(base_type.name, base_type)

    def visit_declaration(self, node):
        self.add_symbol(node)
        # the members inside struct are a new scope
        base_type = node.get_base_type()
        if node.name is not None and isinstance(base_type, StructType):
            if isinstance(node.type, StructType) or isinstance(node.type, ArrayType):
                struct_str = 'struct_' + node.name
                self.push_table(node, struct_str, STRUCTTYPE)
                struct_symbol = self.current_scope.lookup_struct(base_type.name)
                if struct_symbol is None:
                    comment = f"Struct '{base_type.name}' not found({node.line})"
                    raise CompilerError(comment)

                struct_symbol.scope_symbol = self.current_scope
                if base_type.exprs.is_null():
                    base_type.exprs = struct_symbol.exprs
                base_type.exprs.visit(self)
                self.pop_table()

        if node.has_initialized():
            init_node = self.assign_init_variables(node, node.init_value)
            node.add_init_value(init_node)
            # now visit its initial values
            node.init_value.visit(self)

    def assign_init_variables(self, decl_node, init_values, scope_name=None):
        # set the init_node to the node
        nodes = NodeList()
        if isinstance(decl_node.type, ArrayType):
            array_index = 0
            for child_node in init_values.nodes:
                sub_node = ArrayOp(Id(decl_node.name, lineno=decl_node.line), Const(array_index, BaseType('int')),
                                   lineno=decl_node.line)
                if scope_name is not None:
                    sub_node = StructOp(scope_name, op='.', right=sub_node, lineno=decl_node.line)
                child_init_node = BinOp(left=sub_node, op='=', right=child_node, lineno=decl_node.line)
                nodes.add(child_init_node)
                array_index += 1
        elif isinstance(decl_node.type, StructType):
            for index in range(len(init_values.nodes)):
                init_value = init_values.nodes[index]
                variable = decl_node.type.exprs.nodes[index]
                init_nodes = self.assign_init_variables(variable, init_value, Id(decl_node.name))
                for child_init_node in init_nodes.nodes:
                    if scope_name is not None:
                        child_init_node = StructOp(scope_name, op='.', right=child_init_node)
                    nodes.add(child_init_node)
        else:
            sub_node = Id(decl_node.name, lineno=decl_node.line)
            if scope_name is not None:
                init_node = BinOp(StructOp(scope_name, op='.', right=sub_node), op='=', right=init_values,
                                  lineno=decl_node.line)
            else:
                init_node = BinOp(left=sub_node, op='=', right=init_values, lineno=decl_node.line)

            nodes.add(init_node)

        return nodes

    def visit_function_type(self, node):
        node.parms.visit(self)

    def visit_parameter_list(self, node):
        """Assign a number to each parameter.  This will later be
           useful for the code generation phase."""
        parms_index = 0
        for parm in node.nodes:
            parm.visit(self)
            parm.parms_index = parms_index
            parms_index += 1

    def visit_function_defn(self, node):
        self.add_symbol(node)
        self.push_table(node, node.name)

        if not node.type.is_null():
            # insert parameters into current symbols
            node.type.visit(self)

        # insert local variables into children
        node.body.visit(self)

        self.pop_table()

    def visit_compound_statement(self, node):
        # because compound statement will use BEGIN and END to create a scope
        self.push_table(node, "compound statements")
        self.enclosing_scope = self.current_scope
        node.declarations.visit(self)
        node.statements.visit(self)
        self.pop_table()

    def visit_expression_statement(self, node):
        node.expr.visit(self)

    def visit_if_statement(self, node):
        node.expr.visit(self)
        node.then_stmt.visit(self)
        node.else_stmt.visit(self)

    def visit_return_statement(self, node):
        node.expr.visit(self)

    def visit_for_loop(self, node):
        node.begin_stmt.visit(self)
        node.expr.visit(self)
        node.end_stmt.visit(self)
        node.stmt.visit(self)


###############################################################################
#                                                                             #
#  FLOW CONTROL                                                               #
#                                                                             #
###############################################################################


class FlowControlVisitor(NodeVisitor):
    """Performs flow control checking on the AST. This makes sure
    that functions return properly through all branches, that
    break/continue statements are only present within loops, and so
    forth."""
    def __init__(self):
        self.in_loop = False
        self.returns = []
        self.cur_loop = []
        self.curr_func_name = ''
        self.curr_return_line = sys.maxsize

    def visit_empty_node(self, node):
        node.has_return = False

    def visit_statements_list(self, node):
        node.has_return = False
        for stmt in node.nodes:
            if node.has_return:
                comment = f"Statements starting at line{stmt.line} is unreachable in '{self.curr_func_name}' "
                issue_collector.add(WarningIssue(comment))
                # now we need judge that we can prevent generating codes for the rest parts
                stmt.set_to_ignore()

            stmt.visit(self)
            if stmt.has_return:
                node.has_return = True
                # obtain the statement with the smallest line, actually the first time
                if stmt.line < self.curr_return_line:
                    self.curr_return_line = stmt.line

    def visit_declaration_list(self, node):
        for child_node in node.nodes:
            if child_node.is_used and child_node.line > self.curr_return_line:
                child_node.is_used = False

    def visit_translation_unit(self, node):
        self.visit_list(node.nodes)

    def visit_for_loop(self, node):
        """whether it is already in a loop"""
        node.in_func_name = self.curr_func_name
        self.cur_loop.append(self.in_loop)
        self.in_loop = True
        node.stmt.visit(self)
        node.has_return = node.stmt.has_return
        self.in_loop = self.cur_loop.pop()

    def visit_break_statement(self, node):
        if not self.in_loop:
            comment = f"Break statement outside of loop({node.line})."
            raise CompilerError(comment)

    def visit_continue_statement(self, node):
        if not self.in_loop:
            comment = f"Continue statement outside of loop({node.line})."
            raise CompilerError(comment)

    def visit_if_statement(self, node):
        node.in_func_name = self.curr_func_name
        node.expr.visit(self)
        node.then_stmt.visit(self)
        node.else_stmt.visit(self)
        if node.then_stmt.has_return and node.else_stmt.has_return:
            node.has_return = True
        elif (node.expr.is_const() and int(node.expr.expr) > 0) and node.then_stmt.has_return:
            # if (1) then return, is definitely a return
            node.has_return = True

    def visit_function_defn(self, node):
        self.curr_func_name = node.name
        self.in_loop = False
        node.body.visit(self)
        if not node.return_type.get_outer_string() == 'void' and not node.body.has_return:
            comment = f"Function '{self.curr_func_name}' doesn't return through all branches."
            raise CompilerError(comment)
        elif node.return_type.get_outer_string() == 'void' and node.body.has_return:
            comment = f"Function '{self.curr_func_name}' return values while it is void."
            raise CompilerError(comment)
        # determine whether function has the ignore parts
        if len(self.returns) > 1:
            node.has_ignore_parts = True
            self.returns[-1].is_final_one = True

    def visit_return_statement(self, node):
        node.has_return = True
        node.in_func_name = self.curr_func_name
        self.returns.append(node)

    def visit_compound_statement(self, node):
        self.curr_return_line = sys.maxsize
        node.statements.visit(self)
        node.has_return = node.statements.has_return
        node.declarations.visit(self)


###############################################################################
#                                                                             #
#  TYPE CHECKING                                                              #
#                                                                             #
###############################################################################


class TypeCheckVisitor(NodeVisitor):
    """Visitor that performs type checking on the AST, attaching a
    Type object subclass to every eligible node and making sure these
    types don't conflict."""

    def __init__(self):
        self.curr_func = None

    @staticmethod
    def process_condition(expr):
        """Does simple type checking for an expression."""
        if expr.type.get_outer_string() not in ('int', 'char'):
            if expr.type.get_outer_string() is 'pointer':
                # the expr has pointer comparison
                pass
            else:
                comment = f"Conditional expression is '{expr.type.get_outer_string()}', "\
                          f"which doesn't evaluate to an int/char."
                raise CompilerError(comment)

    @staticmethod
    def coerce_const(var, _type):
        """If the given typed terminal is a constant, coerces it to
         the given type."""
        if var.is_const() and _type.get_string() in ('int', 'char'):
            var.type = _type

    def coerce_consts(self, lhs, rhs):
        """Looks at two typed terminals to see if one of them
        is a constant integral.  If it is, then coerce it to
        the type of the other terminal."""
        if lhs.is_const():
            self.coerce_const(lhs, rhs.type)
        elif rhs.is_const():
            self.coerce_const(rhs, lhs.type)

    @staticmethod
    def compare_types(name_str, from_type, to_type, lineno):
        """Compares the two types to see if it's possible to perform a
        binary operation on them."""
        conflict = 0
        from_str = from_type.get_string()
        to_str = to_type.get_string()
        if from_str != to_str:
            if from_str == 'char':
                if to_str == 'int':
                    pass
                else:
                    conflict = 2
            elif from_str == 'int':
                if to_str == 'char':
                    conflict = 1
            else:
                sub_from_str = from_type.get_outer_string()
                sub_to_str = to_type.get_outer_string()
                # allow cast
                if sub_from_str != sub_to_str:
                    conflict = 2

        if conflict == 1:
            comment = f"{name_str}: Conversion from '{from_str}' to '{to_str}' may result in data loss({lineno})."
            issue_collector.add(WarningIssue(comment))
        elif conflict == 2:
            comment = f"{name_str}: Cannot convert from '{from_str}' to '{to_str}'({lineno})."
            raise CompilerError(comment)

    def visit_ast(self, node):
        pass

    def visit_empty_node(self, node):
        node.type = BaseType('void')

    def visit_id(self, node):
        node.type = node.symbol.type

    def visit_negative(self, node):
        node.expr.visit(self)
        node.type = node.expr.type

    def visit_addr_of(self, node):
        node.expr.visit(self)
        if not node.expr.is_lvalue():
            comment = f"Address-of (&) target has no address!({node.line})"
            raise CompilerError(comment)
        else:
            node.expr.set_oaddr()
            node.type = PointerType(node.expr.type)

    def visit_pointer(self, node):
        node.expr.visit(self)
        if node.expr.type.get_outer_string() == 'pointer':
            # add pointer, iterate
            node.type = node.expr.type.child
            node.set_lvalue()
        else:
            comment = f"Pointer dereference (*) target is not a pointer!({node.line})"
            raise CompilerError(comment)

    def visit_struct_op(self, node):
        # use the member type instead
        node.parent.visit(self)
        # determine the current operator
        if isinstance(node.parent.type, PointerType) and not node.op == '->':
            comment = f"Access the point struct member using '.'({node.line})"
            issue_collector.add(ErrorIssue(comment))
        elif isinstance(node.parent.type, StructType) and not node.op == '.':
            comment = f"Access the struct member using '->'({node.line})"
            issue_collector.add(ErrorIssue(comment))

        node.expr.visit(self)
        node.type = node.expr.type
        # lvalue control
        if node.expr.is_lvalue():
            node.set_lvalue()

    def visit_bin_op(self, node):
        node.left.visit(self)
        node.right.visit(self)
        if node.op == '=':
            if not node.left.is_lvalue():
                comment = f"'{node.left.expr}' is an invalid lvalue: not an address!"
                raise CompilerError(comment)
            if isinstance(node.left, Pointer):
                node.left.set_oaddr()
            self.coerce_const(node.right, node.left.type)
            self.compare_types("Assignment", node.right.type, node.left.type, node.line)
            node.type = node.left.type
        else:
            # specification for binary operand type coercion.
            self.coerce_consts(node.left, node.right)
            self.compare_types("BinOp '%s'" % node.op, node.right.type, node.left.type, node.line)
            node.type = node.left.type

    def visit_node_list(self, node):
        self.visit_list(node.nodes)

    def visit_compound_statement(self, node):
        # since we need to check there are initial values set
        node.declarations.visit(self)
        node.statements.visit(self)

    def visit_declaration(self, node):
        # 1. determine the lvalue
        node.set_lvalue()
        # 2. prevent void a;
        if node.type.get_outer_string() == 'void':
            comment = f"Cannot declare '{node.name}' with void type({node.line})"
            raise CompilerError(comment)

        if node.has_initialized():
            node.init_value.visit(self)

    def visit_array_op(self, node):
        node.expr.visit(self)
        node.index.visit(self)
        if node.index.type.get_outer_string() not in ('int', 'char'):
            comment = f"Array index is not an int or char!({node.line})"
            raise CompilerError(comment)
        elif node.expr.type.get_outer_string() != 'pointer':
            comment = f"Array expression is not a pointer!({node.line})"
            raise CompilerError(comment)
        else:
            node.type = node.expr.type.child
            # mark it is a valid left value
            node.set_lvalue()

    def visit_function_op(self, node):
        node.expr.visit(self)
        if not isinstance(node.expr.type, FunctionType):
            comment = f"Target of function expression is not a function!({node.line})"
            raise CompilerError(comment)
        node.type = node.expr.symbol.type.get_return_type()
        node.args.visit(self)
        parms = node.expr.symbol.type.parms
        num_args = len(node.args.nodes)
        num_parms = len(parms.nodes)
        if num_args > num_parms:
            comment = f"Too many arguments passed to function.({node.line})"
            raise CompilerError(comment)
        elif num_args < num_parms:
            comment = f"Too few arguments passed to function.({node.line})"
            raise CompilerError(comment)
        for arg, parm in zip(node.args.nodes, parms.nodes):
            self.coerce_const(arg, parm.type)
            self.compare_types("Function call argument", arg.type, parm.type, node.line)

    def visit_function_defn(self, node):
        self.curr_func = node
        node.body.visit(self)

    def visit_return_statement(self, node):
        node.expr.visit(self)
        return_type = self.curr_func.return_type
        self.coerce_const(node.expr, return_type)
        self.compare_types("Return expression", node.expr.type, return_type, node.line)
        node.expr.coerce_to_type = return_type

    def visit_if_statement(self, node):
        node.expr.visit(self)
        # process_condition is necessary
        self.process_condition(node.expr)
        node.then_stmt.visit(self)
        node.else_stmt.visit(self)

    def visit_for_loop(self, node):
        node.begin_stmt.visit(self)
        node.end_stmt.visit(self)
        node.expr.visit(self)
        # support for(;;)
        if not node.expr.is_null():
            self.process_condition(node.expr)
        node.stmt.visit(self)

    def visit_expression_statement(self, node):
        node.expr.visit(self)


###############################################################################
#                                                                             #
#  SYNTAX CHECKING                                                            #
#                                                                             #
###############################################################################


class Syntax:
    def __init__(self, ast_tree):
        self.ast_tree = ast_tree

    # syntax check
    def check(self):
        # 1. scope check
        scope_check = SymbolTableVisitor()
        with compiler_error_protect():
            scope_check.visit(self.ast_tree)

        if not issue_collector.ok():
            return None

        # 2. flow check
        flow_check = FlowControlVisitor()
        with compiler_error_protect():
            flow_check.visit(self.ast_tree)

        if not issue_collector.ok():
            return None

        # 3. type check
        type_check = TypeCheckVisitor()
        with compiler_error_protect():
            type_check.visit(self.ast_tree)

        if not issue_collector.ok():
            return None

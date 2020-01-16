###############################################################################
#  AST visualizer - generates a DOT file for Graphviz.                        #
#                                                                             #
#  To generate an image from the DOT file run $ dot -Tpng -o ast.png ast.dot  #
#                                                                             #
###############################################################################
import textwrap
from cparser import *


class ASTVisualizer(NodeVisitor):
    def __init__(self, parser):
        self.parser = parser
        self.count = 1
        self.dot_header = [textwrap.dedent("""\
        digraph astgraph {
          node [shape=circle, fontsize=12, fontname="Courier", height=.1];
          ranksep=.3;
          edge [arrowsize=.5]
        """)]
        self.dot_body = []
        self.dot_footer = ['}']

    def visit_ast(self, node):
        # noting to de
        pass

    def visit_empty_node(self, node):
        s = ' node{} [label="Empty"]\n'.format(self.count)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1

    def visit_node_list(self, node):
        for child_node in node.nodes:
            child_node.visit(self)
            s = '  node{} -> node{}\n'.format(node._num, child_node._num)
            self.dot_body.append(s)

    def visit_translation_unit(self, node):
        s = ' node{} [label="Program"]\n'.format(self.count)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1
        # visit the children
        self.visit_node_list(node)

    def visit_function_defn(self, node):
        s = ' node{} [label="{}"]\n'.format(self.count, node.name)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1
        # visit the function, return_type, and body)
        for child_node in (node.return_type, node.function, node.body):
            child_node.visit(self)
            s = '  node{} -> node{}\n'.format(node._num, child_node._num)
            self.dot_body.append(s)

    def visit_compound_statement(self, node):
        s = ' node{} [label="Statement"]\n'.format(self.count)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1
        # visit the children
        for child_node in (node.declarations, node.statements):
            child_node._num = node._num
            self.visit_node_list(child_node)

    def visit_return_statement(self, node):
        s = ' node{} [label="Return"]\n'.format(self.count)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1

        node.expr.visit(self)
        s = '  node{} -> node{}\n'.format(node._num, node.expr._num)
        self.dot_body.append(s)

    def visit_if_statement(self, node):
        s = ' node{} [label="If"]\n'.format(self.count)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1
        # visit the conditions
        node.expr.visit(self)
        s = '  node{} -> node{}\n'.format(node._num, node.expr._num)
        self.dot_body.append(s)
        # visit the statements
        for child_node in (node.then_stmt, node.else_stmt):
            child_node.visit(self)
            s = '  node{} -> node{}\n'.format(node._num, child_node._num)
            self.dot_body.append(s)

    def visit_declaration(self, node):
        s = ' node{} [label="{}"]\n'.format(self.count, node.name)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1

        node.type.visit(self)
        s = '  node{} -> node{}\n'.format(node._num, node.type._num)
        self.dot_body.append(s)

    def visit_type(self, node):
        s = ' node{} [label="void"]\n'.format(self.count)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1

    def visit_base_type(self, node):
        s = ' node{} [label="{}"]\n'.format(self.count, node.type_str)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1
        # visit the children
        if not node.child.is_null():
            node.child.visit(self)
            s = '  node{} -> node{}\n'.format(node._num, node.child._num)
            self.dot_body.append(s)

    def visit_function_type(self, node):
        s = ' node{} [label="Parameters"]\n'.format(self.count)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1
        # visit the children
        node.parms._num = node._num
        self.visit_node_list(node.parms)

    def visit_pointer_type(self, node):
        s = ' node{} [label="*"]\n'.format(self.count)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1
        # visit the children
        node.child.visit(self)
        s = '  node{} -> node{}\n'.format(node._num, node.child._num)
        self.dot_body.append(s)

    def visit_array_type(self, node):
        # visit the left and right hands
        for child_node in (node.index, node.child):
            child_node.visit(self)
            s = '  node{} -> node{}\n'.format(node._num, child_node._num)
            self.dot_body.append(s)

    def visit_bin_op(self, node):
        s = ' node{} [label="{}"]\n'.format(self.count, node.op)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1
        # visit the left and right hands
        for child_node in (node.left, node.right):
            child_node.visit(self)
            s = '  node{} -> node{}\n'.format(node._num, child_node._num)
            self.dot_body.append(s)

    def visit_array_op(self, node):
        s = ' node{} [label="[]"]\n'.format(self.count)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1

        # visit the left and right hands
        for child_node in (node.expr, node.index):
            child_node.visit(self)
            s = '  node{} -> node{}\n'.format(node._num, child_node._num)
            self.dot_body.append(s)

    def visit_function_op(self, node):
        node.expr.visit(self)
        node._num = node.expr._num
        # visit the arguments
        node.args._num = node._num
        self.visit_node_ist(node.args)

    def visit_struct_op(self, node):
        node.parent.visit(self)
        node._num = node.parent._num

        node.expr.visit(self)
        s = '  node{} -> node{}\n'.format(node._num, node.expr._num)
        self.dot_body.append(s)

    def visit_id(self, node):
        s = ' node{} [label="{}"]\n'.format(self.count, node.expr)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1

    def visit_const(self, node):
        s = ' node{} [label="{}"]\n'.format(self.count, node.value)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1

    def visit_addr_of(self, node):
        s = ' node{} [label="&"]\n'.format(self.count)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1

        node.expr.visit(self)
        s = '  node{} -> node{}\n'.format(node._num, node.expr._num)
        self.dot_body.append(s)

    def visit_pointer(self, node):
        s = ' node{} [label="*"]\n'.format(self.count)
        self.dot_body.append(s)
        node._num = self.count
        self.count += 1

        node.expr.visit(self)
        s = '  node{} -> node{}\n'.format(node._num, node.expr._num)
        self.dot_body.append(s)

    def gen_tree(self):
        tree = self.parser.parse()
        self.visit(tree)
        return ''.join(self.dot_header + self.dot_body + self.dot_footer)


def main():
    file_name = './c_test_suite/000' + str(17) + '.c'
    with open(file_name, 'r') as f:
        text = f.read()

    lexer = Lexer(text)
    parser = Parser(lexer)
    viz = ASTVisualizer(parser)
    content = viz.gen_tree()

    file_name = 'ast.dot'
    with open(file_name, 'w') as f:
        f.write(content)

    # os.system('dot -Tpng -o ast.png {}'.format(file_name))


if __name__ == '__main__':
    main()


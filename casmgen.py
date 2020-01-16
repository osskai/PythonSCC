# -*- coding: utf-8 -*-
from cframe import *
from csyntax import *


class RegistersManager:
    """This class attempts to abstract the x86-64 registers into a 'stack
    machine'.  Calling push() gives you a register that isn't currently
    in use by the stack machine, pop() gives you a register with the
    value of the most recently pushed element.

    the stack can manager all compile_loc besides the registers
    """

    def __init__(self, parent):
        # The parent CodeGenVisitor object of this stack machine.
        self.parent = parent

        # A list of all registers on the machine.
        self.all_regs = [Rax, Rbx, Rcx, Rdx, Rsi, Rdi, R8, R9, R10, R11]

        # A list of the registers currently free.
        self.regs_free = self.all_regs[:]

        # A list of all the registers that are "almost" free
        self.regs_almost_free = []

        # A list of all the temporary variable memory locations that are currently unused.
        self.mem_free = []

        # A list corresponding to the actual stack of the stack
        # machine.  The item at the top of the stack is the
        # last element of this list.
        self.stack = []

        # The location of the next memory location to be used for
        # temporary variables, relative to the current function's
        # frame pointer.
        # the lower space for temporarily use
        self.next_temp = 0

        # A list of the callee-save registers that have been used
        # so far by this function.  Once processing is finished,
        # these registers will be pushed onto the process' stack
        # at the beginning of the function and popped off just
        # before the function terminates.
        self.callee_save_regs_used = []

        # A list of the caller-save registers on the machine.
        self.caller_save_regs = [R10, R11]

        # A list of the callee-save registers on the machine.
        self.callee_save_regs = [Rbx, Rbp]

    def set_base_fp(self, base_fp):
        self.next_temp = base_fp - WORD_SIZE

    def output(self, op, left_reg, right_reg=None):
        self.parent.output(op, left_reg, PointerType(), right_reg)

    def save_caller_saves(self):
        """Saves the caller-save registers, which should be done
        before the current function makes a function call, so that
        the registers don't get corrupted by the called function."""
        for reg in self.caller_save_regs:
            if reg not in self.regs_free:
                self.copy_reg_to_temp(reg)
                # because we add them here, if we free the regs_free,
                # it is equal to load_caller_saves, like popl %rdx
                self.regs_free.append(reg)

    def save_callee_saves(self):
        """Emits code that pushes the callee-save registers used by
        the stack machine onto the process' stack."""
        # if function does not change the values in these registers, no need to store them
        for reg in self.callee_save_regs_used:
            self.output(Push, reg)

    def load_callee_saves(self):
        """Emits code that pops the callee-save registers used by
        the stack machine off the process' stack."""
        # if reg is not in callee_save_regs_used, they are used by like
        for reg in self.callee_save_regs_used:
            self.output(Pop, reg)

    def copy_reg_to_temp(self, valid_reg):
        """Copy the least recently used register on the stack into a
        temporary variable.  The register must be in the valid_regs
        list."""
        # if no free temp variables exist, create a new one.
        if len(self.mem_free) == 0:
            self.mem_free.append(MemoryFrame(f"(%rsp)", self.next_temp))
            self.next_temp -= WORD_SIZE

        # get an unused temp variables
        mem = self.mem_free.pop()

        # find the least recently used register on the stack
        find_reg = False
        for index, item in enumerate(self.stack):
            if item == valid_reg:
                # Modify the element's stack machine position to reflect its new location.
                self.stack[index] = mem
                find_reg = True
                break
        if not find_reg:
            comment = f"No free registers inside OR outside of stack!"
            raise CompilerError(comment)

        # emit code to copy the register to the memory location.
        self.output(Mov, valid_reg, mem)

    def get_free_reg(self, valid_regs=None, preferred_reg=None):
        """Returns a free register that is in the valid_regs list.  If
        no registers are available, the most least-recently used
        eligible one is freed."""
        if valid_regs is None:
            valid_regs = self.all_regs

        # If we have a register free, return it.
        if len(self.regs_free) > 0:
            reg = None
            if preferred_reg is not None and preferred_reg in self.regs_free:
                # follow the use rule for the registers
                reg = preferred_reg
            else:
                for item in self.regs_free:
                    if item in valid_regs:
                        reg = item
                        break

            if reg is not None:
                self.remove_free_reg(reg)
                return reg

        # copy the first register into a temp variable and return the register.
        return self.copy_reg_to_temp(valid_regs[0])

    def remove_free_reg(self, reg):
        """pop caller-save register"""
        self.regs_free.remove(reg)
        # If this register is a callee-save register that
        # we haven't used before, add it to our list
        # of used callee-save registers.
        if reg in self.callee_save_regs and reg not in self.callee_save_regs_used:
            self.callee_save_regs_used.append(reg)

    def validate(self, preferred_reg=None):
        """Finds a free eligible register (or frees one if all are
        being used) and returns it, but do not pushing it onto the stack"""
        reg = self.get_free_reg(self.all_regs, preferred_reg)

        return reg

    def push(self, preferred_reg=None, is_reg=True, valid_regs=None):
        """Finds a free eligible register (or frees one if all are
        being used) and returns it, pushing the register onto the
        stack machine's stack.

        If preferred_reg is passed, this function will try its
        best to return preferred_reg, if it's available."""
        if is_reg:
            if valid_regs is None:
                valid_regs = self.all_regs

            reg = self.get_free_reg(valid_regs, preferred_reg)
        else:
            reg = preferred_reg

        self.stack.append(reg)
        return reg

    def pop(self, valid_regs=None):
        """Pops the top element off the stack machine's stack, coerces
        it to the given type if necessary, and returns a register in
        which the element's value now resides."""
        if valid_regs is None:
            valid_regs = self.all_regs

        return self._pop(valid_regs)

    def _pop(self, valid_regs):
        """Pops the top element of the stack into a free register
        that is also in valid_regs and returns the register name.  If
        no registers are free, the least recently used one is first
        copied into a temporary variable and then used."""
        if self.is_empty():
            comment = f"Internal Error! Report fault to the author!"
            raise CompilerError(comment)

        loc = self.stack.pop()

        # if loc is not a register
        if not loc.is_register():
            return loc

        # If the top of the stack is a register, just return the
        # name of the register and add the register to our free
        # register list.
        if loc in valid_regs:
            self.regs_almost_free.append(loc)
            return loc

        # Otherwise, copy the temp variable at the top of the stack
        # into a free register, possibly requiring us to spill the
        # current contents of the memory register into another temp
        # variable.
        reg = self.get_free_reg(valid_regs)
        self.output(Mov, loc, reg)

        # if our location was a register but not in valid_regs,
        # make the register free for use.
        if loc in self.all_regs:
            self.regs_free.append(loc)

        self.regs_almost_free.append(reg)
        return reg

    def is_empty(self):
        """Returns whether the stack machine is empty."""
        return len(self.stack) == 0

    def peek(self):
        """Returns the top element of the stack, but doesn't pop it.  """
        return self.stack[-1]

    def done(self):
        """Frees all registers that are marked as being in
        intermediate use (i.e., have been pop()'d)."""
        self.regs_free.extend(self.regs_almost_free)
        # sort the regs then for recursively use
        self.regs_free.sort(key=lambda item: item.key)
        self.regs_almost_free = []

    def get_max_fp(self):
        """Returns the maximum point in the process' stack, relative
        to the current function's frame pointer, that the stack
        machine is using for temporary variables."""
        return self.next_temp + WORD_SIZE

    def last_is_memory(self):
        """Whether the register in stack top is a pushed memory: (%rbp)"""
        if self.is_empty():
            return False

        return self.peek().is_memory()


class ASMCode:
    """Stores the ASM code"""
    def __init__(self):
        self.comm = []
        self.data = []
        self.texts = []
        self.string_literals = []

    def add_comm(self, name, size):
        _str = f"  .comm {name}, {size}"
        _str += "{:>35}".format(f"## @{name[1:]}")
        self.comm.append(_str)

    def add_global(self, name):
        _str = f"  .globl {name}"
        _str += "{:>35}".format(f"## -- Begin function {name[1:]}")
        self.texts.append(_str)
        self.texts.append(f"  .p2align 4, 0x90")

    def add_data(self, name, type_str, init_value=None):
        self.data.append(f"{name}:")
        if init_value is not None:
            self.data.append(f"  .{type_str} {init_value}")
        else:
            self.data.append(f"  .zero {4}")

    def add_texts(self, _str):
        self.texts.append(_str)

    def add_string_literal(self, name, chars):
        """Add a string literal to the ASM code."""
        self.string_literals.append(f"{name}:")
        self.string_literals.append(f'  .asciz "{chars}"')

    def output(self):
        """Produce the full assembly code."""
        header = ["  .section \t __TEXT,__text"]
        header += [str(text) for text in self.texts]
        if self.string_literals or self.data:
            header += ["  .section \t __TEXT,__cstring"]
            header += self.data
            header += self.string_literals
            header += [""]
        header += self.comm

        return "\n".join(header)


class CodeGenVisitor(NodeVisitor):
    """Visitor that generates x86 assembly code for the abstract
    syntax tree."""

    def __init__(self, asm_code, show_comments=False, source_code=None):
        """Constructor.  'file' is the file object to output the
        resulting code to.  If 'show_comments' is true, then annotated
        comments are produced for the generated assembly code."""
        NodeVisitor.__init__(self)

        # stack
        self.stack = None

        # store all asm codes here
        self.asm_code = asm_code

        # Whether we should show comments or not.
        self.show_comments = show_comments

        # source code
        self.source_code = source_code
        self.curr_pos = 0

        # current asm code
        self.curr_str = ""

        # The current label number we're on, for generating
        # jump labels in the assembly code (e.g., 'LO', 'L1', etc).
        self.jmp_label = 0
        self.break_labels = []
        self.continue_labels = []

        # Current label number for generating string literal labels.
        self.str_literal_label = 0

        # selection labels
        self.if_body_label = ""
        self.if_else_label = ""
        self.has_body_label = False
        self.has_ignore_parts = False

        # ops
        self.relation_post_ops = []

    def new_label(self):
        """Generate a new jump label and return it."""
        label = f".L{self.jmp_label}"
        self.jmp_label += 1
        return label

    def output(self, op, left_reg, left_type, right_reg=None, right_type=None):
        """Output a line of assembly code to the output file, with an optional annotated comment."""
        # determine the suffix of operation
        _str = f"\t{op.to_str(left_type, right_type)} {left_reg.to_str(left_type)}"
        if right_reg is not None:
            if right_type is not None:
                _str += f", {right_reg.to_str(right_type)}"
            else:
                _str += f", {right_reg.to_str(left_type)}"

        self.curr_str += _str + "\n"

    def comment(self, _str):
        self.curr_str += "\t" + _str + "\n"

    def label_comment(self, _str):
        self.curr_str += _str + "\n"

    def comment_by_line(self, pos):
        # show comments for each line
        if self.show_comments is False:
            return

        while self.curr_pos < pos-1:
            self.comment_by_line(pos - 1)

        self.curr_pos = pos
        # the line is got from the source codes
        line = self.source_code.get(pos-1)
        if line is not None:
            del self.source_code[pos-1]
            comment = f"## {pos:{6}}  : {line}"
            self.curr_str += comment

    def calc_var_size(self, _type):
        """Calculate and return the size of the given type, in bytes."""
        type_str = _type.get_outer_string()
        if type_str == 'char':
            return CHAR_SIZE
        elif type_str == 'int':
            return INT_SIZE
        elif type_str == 'pointer':
            return WORD_SIZE
        elif type_str == 'struct':
            last_member = _type.exprs.nodes[-1]
            type_size = self.allocate_and_align_space(last_member.type, -last_member.offset)
            return abs(type_size)
        else:
            comment = f"Unknown type: {type_str}"
            raise CompilerError(comment)

    def visit_ast(self, node):
        pass

    def visit_translation_unit(self, node):
        """Outputs the entire assembly source file."""
        self.curr_str = ""
        for symbol in node.scope_symbol.symbols.vars.values():
            symbol.compile_loc = '_' + symbol.name
            if isinstance(symbol.type, FunctionType):
                self.curr_str = ""
                self.asm_code.add_global(symbol.compile_loc)
                symbol.visit(self)
                # add the .text
                self.label_comment("{:>35}".format(f"## -- End function"))
                self.asm_code.add_texts(self.curr_str)
            else:
                self.asm_code.add_comm(symbol.compile_loc, self.calc_var_size(symbol.type))
                symbol.compile_loc = GlobalVariableFrame(symbol.compile_loc + f"(%rip)")

    def visit_const(self, node):
        """for int k = 10 + i, we should create a register for 10 to collect the result."""
        left_reg = ImmediateFreme(f"${node.expr}")
        right_reg = self.stack.push()
        self.output(Mov, left_reg, node.type, right_reg)

    def visit_id(self, node):
        if self.stack.last_is_memory():
            # unnecessary move from one register to another register
            if not node.symbol.compile_loc.is_register() and not isinstance(node.type, ArrayType):
                reg = self.stack.push()
                if node.is_oaddr:
                    self.output(Lea, node.symbol.compile_loc, node.type, reg)
                else:
                    self.output(Mov, node.symbol.compile_loc, node.type, reg)

                return

        if node.symbol.compile_loc.is_global():
            mem_reg, _ = self.replace_last_reg_with_memory(node.symbol.compile_loc, PointerType(), 0)
            self.stack.push(mem_reg, is_reg=False)
        else:
            self.stack.push(node.symbol.compile_loc, is_reg=False)

    def replace_last_reg_with_memory(self, compile_loc, reg_type, offset):
        reg = self.stack.push()
        self.output(Mov, compile_loc, reg_type, reg)
        mem_reg = MemoryFrame(f"({reg.to_str(reg_type)})", offset)
        self.stack.pop()

        return mem_reg, reg

    def visit_and_pop(self, node):
        """"""
        if node.is_const():
            return ImmediateFreme(f"${node.expr}")
        else:
            node.visit(self)
            return self.stack.pop()

    def visit_bin_op(self, node):
        """consider the codes:
        int i = 10;         movl $10, -20(%rbp)
        int j = 20;         movl $20, -16(%rbp)
        int k = i + j;      movl -16(%rbp), %eax  # use a temp register
                            addl -20(%rbp), %eax
                            movl %eax, -12(%rbp)
        """
        if node.op == '=':
            self.binop_assign(node)
        elif node.op in ('+', '-', '*'):
            self.binop_arith(node)
        elif node.op in ('==', '!=', '<', '>', '<=', '>='):
            self.binop_compare(node)
        elif node.op in ('&&', '||'):
            self.binop_logical(node)

    def binop_assign(self, node):
        """Performs an assignment operation (=) on the given BinOp node."""
        node.left.visit(self)
        right_reg = self.visit_and_pop(node.right)
        left_reg = self.stack.pop()

        self.output(Mov, right_reg, node.type, left_reg)
        self.stack.done()

    def binop_arith(self, node):
        """Performs an arithmetic operation (+, -, etc) on the given Binop node."""
        node.left.visit(self)
        right_reg = self.visit_and_pop(node.right)
        left_reg = self.stack.pop()

        if isinstance(node.left.type, PointerType) and not isinstance(node.right.type, PointerType):
            left_hand_size = self.calc_var_size(node.left.type.get_base_type())
            left_hand_reg = ImmediateFreme(f"${left_hand_size}")
            if not right_reg.is_immediate():
                if right_reg.is_memory():
                    replaced_reg = self.stack.push()
                    # use PointerType() to prevent using cltq
                    self.output(Mov, right_reg, PointerType(), replaced_reg)
                    self.output(Mul, left_hand_reg, PointerType(), replaced_reg)
                    self.stack.pop()
                    right_reg = replaced_reg
                elif right_reg.is_register():
                    self.output(Mul, left_hand_reg, PointerType(), right_reg)
            else:
                right_reg = left_hand_reg

        # we need to consider the right_reg or left_reg, whether one of or both are pointer dereference
        # TODO: here we do not need to change the instruction
        left_reg, right_reg = exchange_reg_to_left(left_reg, right_reg)
        if not left_reg.is_register():
            self.output(Mov, left_reg, node.left.type, self.stack.push())
            left_reg = self.stack.pop()

        self.output(binop_arith_instrs[node.op], right_reg, node.type, left_reg)
        self.stack.done()
        # push the result reg into the stack
        self.stack.push(left_reg)

    def binop_compare(self, node):
        """Performs a comparison operation (>, ==, etc) on the given Binop node."""
        compare_op_pair = {'>': '<=', '>=': '<', '<': '>=', '<=': '>', '==': '!=', '!=': '=='}

        self.relation_post_ops.append(node.op)

        if node.left.is_const():
            left_reg = ImmediateFreme(f"${node.left.expr}")
            self.stack.push(left_reg, is_reg=False)
        else:
            node.left.visit(self)

        self.relation_post_ops.pop()

        right_reg = self.visit_and_pop(node.right)
        left_reg = self.stack.pop()

        # TODO: if changing the regs, it must corresponding to the judge instructions
        flag, left_reg, right_reg = exchange_reg_to_right(left_reg, right_reg)

        node_op = compare_op_pair[node.op] if flag else node.op

        self.output(Comp, left_reg, node.left.type, right_reg)

        if not len(self.relation_post_ops) == 0:
            if self.relation_post_ops[-1] in ('&&', '||'):
                self.relation_post_ops.pop()
                self.relation_post_ops.append(node_op)
            else:
                if right_reg.is_memory():
                    # change to register
                    self.output(Mov, right_reg, node.right.type, self.stack.push())
                    right_reg = self.stack.peek()

                self.output(binop_relation_instrs[node_op], right_reg, BaseType('char'))
                self.output(Movz, right_reg, BaseType('char'), right_reg, node.right.type)
        else:
            # use simple way
            self.comment(f"{binop_reverse_jumps[node_op]} {self.if_else_label}")

        self.stack.done()

    def binop_logical(self, node):
        """represent the && and ||"""
        self.relation_post_ops.append(node.op)
        self.logical_test_hand(node.left)
        if node.op == '&&':
            if not len(self.relation_post_ops) == 0 and isinstance(node.left, BinOp):
                self.comment(f"{binop_reverse_jumps[self.relation_post_ops[-1]]} {self.if_else_label}")
            else:
                self.comment(f"jz {self.if_else_label}")
        elif node.op == '||':
            # take a || b as an example,
            # if a is true, ignore the process of b
            # if a is false, continue the process of b
            if not len(self.relation_post_ops) == 0 and isinstance(node.left, BinOp):
                self.comment(f"{binop_jumps[self.relation_post_ops[-1]]} {self.if_body_label}")
            else:
                self.comment(f"jnz {self.if_body_label}")
            self.has_body_label = True

        if not len(self.relation_post_ops) == 0:
            self.relation_post_ops.pop()

        self.logical_test_hand(node.right)
        if not isinstance(node.right, BinOp):
            self.comment(f"jz {self.if_else_label}")

    def logical_test_hand(self, node):
        node.visit(self)
        if not self.stack.is_empty():
            compare_reg = self.stack.pop()
            if not compare_reg.is_register():
                self.output(Mov, compare_reg, node.type, self.stack.push())
                compare_reg = self.stack.pop()
                # delete the temporary register
                self.stack.done()

            self.output(Test, compare_reg, node.type, compare_reg)

    def visit_node_list(self, node):
        self.visit_list(node.nodes)

    def visit_compound_statement(self, node):
        child_decl_index = 0
        child_stmt_index = 0
        while child_decl_index < len(node.declarations.nodes) and child_stmt_index < len(node.statements.nodes):
            child_decl_node = node.declarations.nodes[child_decl_index]
            child_stmt_node = node.statements.nodes[child_stmt_index]

            if child_decl_node.line <= child_stmt_node.line:
                self.comment_by_line(child_decl_node.line)
                child_decl_node.visit(self)
                child_decl_index += 1
            else:
                """if the statements cannot be reached, ignore them"""
                if not child_stmt_node.is_needed:
                    return

                self.comment_by_line(child_stmt_node.line)
                child_stmt_node.visit(self)
                child_stmt_index += 1

        if child_decl_index < len(node.declarations.nodes):
            for child_node in node.declarations.nodes[child_decl_index:]:
                self.comment_by_line(child_node.line)
                child_node.visit(self)
        elif child_stmt_index < len(node.statements.nodes):
            for child_node in node.statements.nodes[child_stmt_index:]:
                self.comment_by_line(child_node.line)
                child_node.visit(self)

    def visit_if_statement(self, node):
        """"""
        done_label = self.new_label() + "_done" + f"_{node.in_func_name}"
        if not node.else_stmt.is_null():
            else_label = self.new_label() + "_else" + f"_{node.in_func_name}"
        else:
            else_label = done_label

        #
        body_label = self.new_label() + "_body" + f"_{node.in_func_name}"
        self.logical_expression_judge(node.expr, body_label, else_label)

        # visit the then statement
        self.visit_and_empty_stack(node.then_stmt)

        # visit the else statement if it exists
        if not node.else_stmt.is_null():
            self.comment(f"jmp {done_label}")
            self.label_comment(f"{else_label}:")
            self.visit_and_empty_stack(node.else_stmt)

        self.label_comment(f"{done_label}:")

    def logical_expression_judge(self, node, body_label, else_label):
        """deal with a && b"""
        self.if_body_label = body_label
        self.if_else_label = else_label
        self.has_body_label = False
        self.logical_test_hand(node)
        # if the node is a unary node, append a test comment
        if isinstance(node, UnaryOp) or isinstance(node, Const):
            self.comment(f"jz {self.if_else_label}")
        elif isinstance(node, BinOp) and node.op in ('+', '-', '*', '/'):
            self.comment(f"jz {self.if_else_label}")
        else:
            # we need determine whether we need those comments
            if self.has_body_label:
                self.label_comment(f"{body_label}:")
        self.stack.done()

    def visit_and_empty_stack(self, node):
        """Visit the node and then empty the stack machine of the node's return value, if one exists."""
        node.visit(self)
        if not self.stack.is_empty():
            self.stack.pop()
            self.stack.done()
            if not self.stack.is_empty():
                comment = f"PANIC! Register stack isn\'t empty!"
                raise CompilerError(comment)

    def visit_for_loop(self, node):
        """"""
        # we need to know whether there are breaks or continues
        test_label = self.new_label() + "_test" + f"_{node.in_func_name}"
        done_label = self.new_label() + "_done" + f"_{node.in_func_name}"

        # store the current labels
        self.push_loop_labels(break_label=done_label, continue_label=test_label)

        # visit the begin statement, consider the empty status
        node.begin_stmt.visit(self)

        self.label_comment(f"{test_label}:")
        # visit the end statement
        if not node.end_stmt.is_null():
            node.end_stmt.visit(self)

            compare_reg = self.stack.pop()
            self.stack.done()

            self.output(Test, compare_reg, PointerType(), compare_reg)
            self.comment(f"jz {done_label}")

        # visit the body statements
        self.visit_and_empty_stack(node.stmt)
        # adjust the expression
        node.expr.visit(self)

        if not self.has_ignore_parts:
            self.comment(f"jmp {test_label}")

        if not node.end_stmt.is_null():
            self.label_comment(f"{done_label}:")

        self.pop_loop_labels()

    def push_loop_labels(self, break_label, continue_label):
        """Pushes new values of labels to jump to for 'break' and 'continue' statements."""
        self.break_labels.append(break_label)
        self.continue_labels.append(continue_label)

    def pop_loop_labels(self):
        """Restores old values of labels to jump to for 'break' and 'continue' statements."""
        self.break_labels.pop()
        self.continue_labels.pop()

    def visit_break_statement(self, node):
        self.comment(f"jmp {self.break_labels[-1]}")

    def visit_continue_statement(self, node):
        self.comment(f"jmp {self.continue_labels[-1]}")

    def visit_return_statement(self, node):
        return_reg = self.visit_and_pop(node.expr)
        # optimal in the end
        if not return_reg == Rax:
            self.output(Mov, return_reg, node.expr.coerce_to_type, Rax)

        # exit function
        if self.has_ignore_parts and not node.is_final():
            curr_func_end_label = f".L_function_end" + f"_{node.in_func_name}"
            self.comment(f"jmp {curr_func_end_label}")

        self.stack.done()

    def visit_expression_statement(self, node):
        node.expr.visit(self)

    def visit_declaration(self, node):
        # if assigned the initial value when variable is defined
        if node.has_initialized():
            node.init_value.visit(self)

    def visit_function_defn(self, node):
        """Output the assembly code for a function."""
        # sub labels
        assert len(self.break_labels) == 0 and len(self.continue_labels) == 0

        # Create a new stack machine for this function.
        self.stack = RegistersManager(self)

        # functions
        self.label_comment(f"{node.compile_loc}:" + "{:>35}".format(f"## @{node.name}"))
        self.output(Push, Rbp, PointerType())
        self.output(Mov, Rsp, PointerType(), Rbp)

        # Calculate the base size of the stack frame (not including
        # space for the stack machine's temporary variables).
        stack_frame_size = self.calc_function_var_addrs(node.scope_symbol, -self.calc_var_size(PointerType()))

        # the local variables increase downward, the lowest position is the base to place variables temporarily
        self.stack.set_base_fp(stack_frame_size)

        # insert the code later
        previous_str = self.curr_str
        self.curr_str = ""

        # determine whether exists ignored parts in the function
        if node.has_ignore_parts:
            self.has_ignore_parts = True

        # Generate assembly code for the function.
        self.comment_by_line(node.line)
        node.body.visit(self)

        function_str = self.curr_str
        self.curr_str = previous_str

        if not self.stack.get_max_fp() == 0:
            # store local variables on the stack, do not need permanent locations
            # and they disappear after the function returns
            left_reg = ImmediateFreme(f"${-self.stack.get_max_fp()}")
            self.output(Sub, left_reg, PointerType(), Rsp)

        # Save any callee-save registers that may have been used.
        self.stack.save_callee_saves()

        # Add the previously-generated assembly code for the function.
        self.curr_str += function_str

        if self.has_ignore_parts:
            self.label_comment(f"{'.L_function_end' + f'_{node.name}'}:")

        self.comment_by_line(node.end_line)

        # Restore any callee-save registers that may have been used.
        self.stack.load_callee_saves()

        if not self.stack.get_max_fp() == 0:
            self.output(Mov, Rbp, PointerType(), Rsp)

        self.output(Pop, Rbp, PointerType())

        self.comment("retq")

    def calc_function_var_addrs(self, scope_symbol, last_fp_loc):
        """Calculate the addresses of all local variables in the
        function and attach them to their respective symbols in
        the function's symbol table(s)."""
        self.calc_function_arg_addrs(scope_symbol)
        # theoretically, function definition only has one children ==> Compound Statement
        return self.calc_local_var_addrs(scope_symbol.children[0], last_fp_loc)

    def calc_function_arg_addrs(self, scope_symbol):
        # since at most 6 args can be represented by registers
        arg_size = 0
        for symbol in scope_symbol.symbols.vars.values():
            # calculate the arg size based on their type, and their align
            if symbol.parms_index > 5:
                # if arguments over than 6, use stack to store them
                # arguments of the function grow upward,
                arg_size += WORD_SIZE
                symbol.compile_loc = MemoryFrame(f"(%rbp)", arg_size)
            else:
                symbol.compile_loc = Arg_regs[symbol.parms_index]
                self.stack.remove_free_reg(symbol.compile_loc)

            if not symbol.is_used:
                comment = f"function argument '{symbol.name}' is never used."
                issue_collector.add(WarningIssue(comment))

    def allocate_and_align_space(self, var_type, last_fp_loc):
        # align the base address
        next_fp_loc = last_fp_loc - self.calc_var_size(var_type)
        # adjust location for alignment, obtain the next variables
        align = self.calc_var_size(var_type)
        return self.calc_var_align(align, next_fp_loc)

    @staticmethod
    def calc_var_align(align, next_fp_loc):
        if align is None:
            return next_fp_loc

        bytes_overboard = (-next_fp_loc) % align
        if not bytes_overboard == 0:
            last_fp_loc = next_fp_loc - (align - bytes_overboard)
        else:
            last_fp_loc = next_fp_loc

        return last_fp_loc

    def calc_local_var_addrs(self, scope_symbol, last_fp_loc):
        """calculate local variable address in function body"""
        for symbol in scope_symbol.symbols.vars.values():
            if not isinstance(symbol.type, StructType) and not symbol.is_used:
                comment = f"local variable '{symbol.name}' is never used."
                issue_collector.add(WarningIssue(comment))
                continue
            
            if isinstance(symbol.type, ArrayType):
                index = symbol.type.index
                if not index.is_const():
                    comment = f"Cannot allocate space for variable index."
                    raise CompilerError(comment)
                else:
                    var_type = symbol.type.child
                    if isinstance(var_type, StructType):
                        curr_last_fp, first_var_size = self.calc_struct_member_addrs(var_type, 0)
                        last_fp_loc = self.calc_var_align(first_var_size, last_fp_loc)
                        symbol.compile_loc = MemoryFrame(f"(%rbp)", last_fp_loc)
                        if curr_last_fp < 0:
                            last_fp_loc += curr_last_fp
                        var_type_size = abs(curr_last_fp)
                    else:
                        last_fp_loc = self.allocate_and_align_space(var_type, last_fp_loc)
                        var_type_size = self.calc_var_size(var_type)
                        symbol.compile_loc = MemoryFrame(f"(%rbp)", last_fp_loc + var_type_size)

                    # allocate space for the rest elements
                    for i in range(1, int(index.expr)):
                        last_fp_loc -= var_type_size
            elif isinstance(symbol.type, StructType):
                curr_last_fp, first_var_size = self.calc_struct_member_addrs(symbol.type, 0)
                last_fp_loc = self.calc_var_align(first_var_size, last_fp_loc)
                symbol.compile_loc = MemoryFrame(f"(%rbp)", last_fp_loc)
                if curr_last_fp < 0:
                    last_fp_loc += curr_last_fp
            else:
                last_fp_loc = self.allocate_and_align_space(symbol.type, last_fp_loc)
                symbol.compile_loc = MemoryFrame(f"(%rbp)", last_fp_loc + self.calc_var_size(symbol.type))

        max_last_fp = last_fp_loc
        # recursive calculate local variables inside the scope
        for kid in scope_symbol.children:
            if kid.type == STRUCTTYPE:
                continue

            curr_last_fp = self.calc_local_var_addrs(kid, last_fp_loc)
            # max_last_fp is negative, find the maximum one
            if curr_last_fp < max_last_fp:
                max_last_fp = curr_last_fp

        # adjust location for alignment, to keep the stack aligned
        # on a word-sized boundary
        max_last_fp = self.calc_var_align(WORD_SIZE, max_last_fp)

        return max_last_fp

    def calc_struct_member_addrs(self, node, last_fp_loc):
        struct_members = node.exprs
        first_var_size = None
        for child_node in struct_members.nodes:
            if isinstance(child_node.type, ArrayType):
                index = child_node.type.index
                if not index.is_const():
                    comment = f"Cannot allocate space for variable index."
                    raise CompilerError(comment)
                else:
                    var_type = child_node.type.child
                    last_fp_loc = self.allocate_and_align_space(var_type, last_fp_loc)
                    child_node.offset = abs(last_fp_loc + self.calc_var_size(var_type))
                    # allocate space for the rest elements
                    for i in range(1, int(index.expr)):
                        last_fp_loc -= self.calc_var_size(var_type)
            elif isinstance(child_node.type, StructType):
                curr_fp_loc, first_var_size = self.calc_struct_member_addrs(child_node.type, 0)
                last_fp_loc = self.calc_var_align(first_var_size, last_fp_loc)
                child_node.offset = abs(last_fp_loc)
                if curr_fp_loc < 0:
                    last_fp_loc += curr_fp_loc
                var_type = child_node.type
            else:
                var_type = child_node.type
                last_fp_loc = self.allocate_and_align_space(var_type, last_fp_loc)
                child_node.offset = abs(last_fp_loc + self.calc_var_size(child_node.type))

            if first_var_size is None:
                first_var_size = self.calc_var_size(var_type)

        return last_fp_loc, first_var_size

    def visit_function_op(self, node):
        """Generates assembly for calling a function."""
        # If we're using any caller-save registers, free them up.
        self.stack.save_caller_saves()

        # We need to temporarily reverse the order of the function's
        # arguments because we need to push them onto the stack
        # in reverse order.
        node.args.nodes.reverse()

        arg_num = len(node.args.nodes)
        for arg in node.args.nodes:
            arg_reg = self.visit_and_pop(arg)
            if arg_num > 6:
                offset = (8 - arg_num) * WORD_SIZE
                if not offset == 0:
                    right_reg = ImmediateFreme(f"{-offset}(%rsp)")
                else:
                    right_reg = ImmediateFreme(f"(%rsp)")
            else:
                right_reg = Arg_regs[arg_num-1]

            # find the preferred regs
            if arg_num <= 6:
                result_reg = self.stack.validate(right_reg)
                if not result_reg == right_reg:
                    # we must know who use this preferred reg
                    self.output(Mov, right_reg, arg.type, result_reg)

            if isinstance(arg, AddrOf):
                self.output(Lea, arg_reg, arg.expr.type, right_reg)
            else:
                self.output(Mov, arg_reg, arg.type, right_reg)

            arg_num -= 1

        node.args.nodes.reverse()

        self.comment(f"callq {node.expr.symbol.compile_loc}")

        # The function will place its return value in register %eax.
        # So, we'll push a register from the stack and ask it to
        # give us %eax.
        # before push a preferred register, firstly reset the free register set
        self.stack.done()
        result_reg = self.stack.push(preferred_reg=Rax)
        if not result_reg == Rax:
            self.output(Mov, Rax, node.type, result_reg)

        # we need do other way to obtain the stack size
        arg_stack_size = ((len(node.args.nodes) - 6) * WORD_SIZE)

        if arg_stack_size > 0:
            left_reg = ImmediateFreme(f"${arg_stack_size}")
            self.output(Add, left_reg, PointerType(), Rsp)

    def visit_pointer(self, node):
        """"""
        node.expr.visit(self)
        reg_from = self.stack.peek()
        if reg_from.is_register():
            reg_from = MemoryFrame(f"({reg_from.to_str(PointerType())})", 0)

        if node.is_oaddr():
            # sometimes, we do not need to extract it to a new register
            # pointer dereference
            # pop the old one
            self.stack.pop()
            self.stack.push(reg_from, is_reg=False)

    def visit_addr_of(self, node):
        """"""
        node.expr.visit(self)

    def visit_negative(self, node):
        """Negative do two things:
        negl %edx
        do other operation on %edx
        """
        node.expr.visit(self)
        self.output(Neg, self.stack.peek(), node.expr.type)

    def visit_array_op(self, node):
        """"""
        type_size = self.calc_var_size(node.type)
        node.expr.visit(self)
        # determine whether the index is a constant
        if node.index.is_const():
            reg_index = node.index.expr*type_size
            reg_expr = self.stack.peek()
            if reg_expr.is_memory():
                # modify the address
                self.stack.pop()
                # in python, every thing is object
                reg_expr = reg_expr.shift(-reg_index)
                self.stack.push(reg_expr, is_reg=False)
        else:
            node.index.visit(self)
            reg_index = self.stack.pop()

        reg_expr = self.stack.pop()
        if not node.index.is_const():
            if reg_expr.is_register():
                replace_reg = reg_expr
            else:
                replace_reg = self.stack.push()
                self.output(Mov, reg_expr, node.type, replace_reg)

            addr_reg = MemoryFrame(f"({replace_reg},{reg_index},{type_size})", 0)
        else:
            if reg_expr.is_register():
                addr_reg = MemoryFrame(f"({reg_expr})", reg_index)
            else:
                addr_reg = reg_expr

        if node.is_oaddr():
            reg_to = self.stack.push()
            self.stack.done()
            self.output(Lea, addr_reg, node.type, reg_to)
        else:
            self.stack.push(addr_reg, is_reg=False)

    def visit_struct_op(self, node):
        """
        consequent field for struct,
        struct Node {
            struct Node* next;
            int value;
        };
        using offset register, rather than memory
        """
        parent_reg = node.parent.symbol.compile_loc
        if node.op == '->' or (isinstance(node.parent, Pointer) and node.op == '.'):
            signed_offset = 1
            if parent_reg.is_memory():
                parent_reg, replaced_reg = self.replace_last_reg_with_memory(parent_reg, node.type, 0)
                node.parent.symbol.compile_loc = replaced_reg
            else:
                parent_reg = MemoryFrame(f"({parent_reg.to_str(node.type)})", 0)
        else:
            # using signed offset to distinguish register and memory.
            signed_offset = -1

        member_node = node.expr
        if isinstance(member_node, StructOp):
            parent_offset = signed_offset*member_node.parent.symbol.offset
            member_node.parent.symbol.compile_loc = parent_reg.shift(parent_offset)
        elif isinstance(member_node, ArrayOp):
            offset = signed_offset*member_node.expr.symbol.offset
            member_node.expr.symbol.compile_loc = parent_reg.shift(offset)
        else:
            offset = signed_offset*member_node.symbol.offset
            member_node.symbol.compile_loc = parent_reg.shift(offset)

        self.stack.done()
        node.expr.visit(self)

    def visit_string_literal(self, node):
        """string literals
        such as: printf("hello world!");  --> "hello world!"
        """
        label_str = self.get_new_string_literal_label(node.get_str())
        label_reg = GlobalVariableFrame(label_str + f"(%rip)")
        self.output(Lea, label_reg, node.type, self.stack.push())

    def get_new_string_literal_label(self, str_literal):
        label_str = f"Lstr{self.str_literal_label}"
        str_literal = str_literal.replace('\n', '\\12')
        self.asm_code.add_string_literal(label_str, str_literal)
        self.str_literal_label += 1
        return label_str


###############################################################################
#                                                                             #
#  GENERATE x86-64 ASM CODE                                                   #
#                                                                             #
###############################################################################


class GenASM:
    def __init__(self, ast_tree, source_code):
        self.ast_tree = ast_tree
        self.source = source_code

    def mask_asm(self, show_comments=True):
        compiler = CodeGenVisitor(ASMCode(), show_comments, self.source)
        with compiler_error_protect():
            compiler.visit(self.ast_tree)

        return compiler.asm_code.output()

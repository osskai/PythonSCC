# -*- coding: utf-8 -*-

# Size of the 'char' type.
CHAR_SIZE = 1
# Size of the 'int' type.
INT_SIZE = 4
# Size of the 'pointer' type
WORD_SIZE = 8


class Frame:
    """the Frame represents the each token in the asm files"""
    def __init__(self, name):
        self.name = name

    def to_str(self, _type):
        return self.name

    def is_register(self):
        return False

    def is_memory(self):
        return False

    def is_immediate(self):
        return False

    def is_global(self):
        return False

    def __eq__(self, other):
        return self.name == other

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash((self.__class__.__name__, self.name))


class RegisterFrame(Frame):
    """Accept a base register and convert it to right size register
    for example:
        Rax.to_str(node.type) --> '%eax'
    """
    byte_base_regs = ['%rax', '%rbx', '%rcx', '%rdx']
    byte_index_regs = ['%rsi', '%rdi']
    byte_extend_regs = ['%r8', '%r9', '%r10', '%r11']
    stack_regs = ['%rbp', '%rsp']

    def __init__(self, name, key=None):
        Frame.__init__(self, name)
        self.key = key if key is not None else 0

    def to_str(self, _type):
        type_str = _type.get_outer_string()
        if type_str == 'char':
            output_str = self.reg_to_8()
        elif type_str == 'int':
            output_str = self.reg_to_32()
        else:
            output_str = self.name

        if not output_str[0] == '%':
            output_str = '%' + output_str

        return output_str

    def is_register(self):
        return True

    def reg_to_8(self):
        if self.name in self.byte_base_regs:
            return self.name[2] + 'l'
        elif self.name in self.byte_index_regs:
            return self.name[2] + 'il'
        elif self.name in self.byte_extend_regs:
            return self.name[1:] + 'b'
        else:
            raise Exception(f"Register {self.name} is not byte-compatible!")

    def reg_to_32(self):
        if self.name in self.byte_base_regs or self.name in self.byte_index_regs:
            return 'e' + self.name[2:]
        elif self.name in self.byte_extend_regs:
            return 'r' + self.name[2:] + 'd'
        else:
            raise Exception(f"Register {self.name} is not byte-compatible!")


# the known register
Rax = RegisterFrame('%rax', key=0)
Rbx = RegisterFrame('%rbx', key=1)
Rcx = RegisterFrame('%rcx', key=2)
Rdx = RegisterFrame('%rdx', key=3)
Rsi = RegisterFrame('%rsi', key=4)
Rdi = RegisterFrame('%rdi', key=5)
R8  = RegisterFrame('%r8', key=6)
R9  = RegisterFrame('%r9', key=7)
R10 = RegisterFrame('%r10', key=8)
R11 = RegisterFrame('%r11', key=9)
Rbp = RegisterFrame('%rbp', key=10)
Rsp = RegisterFrame('%rsp', key=11)
# argument registers
Arg_regs = [Rdi, Rsi, Rdx, Rcx, R8, R9]


class MemoryFrame(Frame):
    """Represent a memory"""
    def __init__(self, name, offset):
        Frame.__init__(self, name)
        self.offset = offset

    def is_memory(self):
        return True

    def to_str(self, _type):
        if self.offset == 0:
            return self.name

        return str(self.offset) + self.name

    def shift(self, offset):
        cur_offset = self.offset
        cur_offset += offset
        return MemoryFrame(self.name, cur_offset)


class ImmediateFreme(Frame):
    """Represent a direct number"""
    def __init__(self, name):
        Frame.__init__(self, name)

    def is_immediate(self):
        return True


def swap_reg(left_reg, right_reg):
    tmp_reg = left_reg
    left_reg = right_reg
    right_reg = tmp_reg

    return left_reg, right_reg


def sort_regs(left_reg, right_reg):
    if left_reg.is_memory():
        left_reg, right_reg = swap_reg(left_reg, right_reg)
    elif left_reg.is_register():
        if right_reg.is_immediate():
            left_reg, right_reg = swap_reg(left_reg, right_reg)

    return left_reg, right_reg


def exchange_reg_to_left(left_reg, right_reg):
    if right_reg.is_register():
        left_reg, right_reg = swap_reg(left_reg, right_reg)

    return left_reg, right_reg


def exchange_reg_to_right(left_reg, right_reg):
    if left_reg.is_memory() and right_reg.is_memory():
        return False, left_reg, right_reg

    if left_reg.is_immediate():
        return False, left_reg, right_reg
    elif right_reg.is_immediate() and left_reg.is_memory():
        left_reg, right_reg = swap_reg(left_reg, right_reg)
    elif left_reg.is_register() and right_reg.is_memory():
        left_reg, right_reg = swap_reg(left_reg, right_reg)

    return False, left_reg, right_reg


class GlobalVariableFrame(Frame):
    """Represent a global variable or strings"""
    def __init__(self, name):
        Frame.__init__(self, name)

    def is_global(self):
        return True


class OperationFrame(Frame):
    """Represent a operation"""
    compare_ops = ['sete', 'setne', 'setg', 'setge', 'setl', 'setle']
    suffix_str = {'char': 'b', 'int': 'l', 'pointer': 'q'}

    def __init__(self, name):
        Frame.__init__(self, name)

    def to_str(self, _type, add_type=None):
        if self.name in self.compare_ops:
            # do nothing for compare operations
            return self.name
        elif self.name == 'movz':
            # using another way to convert this operation
            from_str = self.suffix_str[_type.get_outer_string()]
            to_str = self.suffix_str[add_type.get_outer_string()]
            if from_str is not None and to_str is not None:
                return 'movz' + from_str + to_str
            else:
                raise NotImplementedError('unexpected type')

        type_str = _type.get_outer_string()
        if type_str not in self.suffix_str.keys():
            type_str = 'pointer'

        output_str = self.suffix_str[type_str]
        if output_str is not None:
            output_str = self.name + output_str
        else:
            raise NotImplementedError('unexpected type')

        return output_str


# the known operation
Mov = OperationFrame('mov')
Movz = OperationFrame('movz')

Add = OperationFrame('add')
Sub = OperationFrame('sub')
Mul = OperationFrame('imul')
Div = OperationFrame('idiv')
binop_arith_instrs = {'+': Add, '-': Sub, '*': Mul, '/': Div}

Push = OperationFrame('push')
Pop = OperationFrame('pop')
Lea = OperationFrame('lea')
Neg = OperationFrame('neg')
Comp = OperationFrame('cmp')
Test = OperationFrame('test')
ShiftLeft = OperationFrame('shl')

Je = OperationFrame('je')
Jne = OperationFrame('jne')
Jg = OperationFrame('jg')
Jge = OperationFrame('jge')
Jl = OperationFrame('jl')
Jle = OperationFrame('jle')
binop_reverse_jumps = {'==': Jne, '!=': Je, '>': Jle, '>=': Jl, '<': Jge, '<=': Jg}
binop_jumps = {'==': Je, '!=': Jne, '>': Jg, '>=': Jge, '<': Jl, '<=': Jle}

SetEq = OperationFrame('sete')
SetNEq = OperationFrame('setne')
SetG = OperationFrame('setg')
SetGEq = OperationFrame('setge')
SetL = OperationFrame('setl')
SetLEq = OperationFrame('setle')
binop_relation_instrs = {'==': SetEq, '!=': SetNEq, '>': SetG, '>=': SetGEq, '<': SetL, '<=': SetLEq}



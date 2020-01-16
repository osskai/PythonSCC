  .section 	 __TEXT,__text
  .globl _foo           ## -- Begin function foo
  .p2align 4, 0x90
_foo:                            ## @foo
	pushq %rbp
	movq %rsp, %rbp
	subq $8, %rsp
##      1  : int foo(int a, int b)
##      2  : {
##      3  :     return a + b;
	addl %edi, %esi
	movl %esi, %eax
##      4  : }
	movq %rbp, %rsp
	popq %rbp
	retq
                 ## -- End function

  .globl _main          ## -- Begin function main
  .p2align 4, 0x90
_main:                           ## @main
	pushq %rbp
	movq %rsp, %rbp
	subq $16, %rsp
##      5  : 
##      6  : int main()
##      7  : {
##      8  :     int a;
##      9  :     a = 1;
	movl $1, -8(%rbp)
##     10  :     return foo(a, 2);
	movl $2, %esi
	movl -8(%rbp), %edi
	callq _foo
.L_function_end_main:
##     11  : }
	movq %rbp, %rsp
	popq %rbp
	retq
                 ## -- End function

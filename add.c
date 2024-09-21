#include "blis.h"
#include STRINGIFY_INT(../PASTEMAC(plugin,BLIS_PNAME_INFIX).h)
#include "bli_x86_asm_macros.h"

void bli_daxpys_mxn(
	const dim_t m,
	const dim_t n0,
	double* restrict alpha,
    double* restrict x,
    const inc_t rs_x0,
    const inc_t cs_x0,
    double* restrict beta,
    double* restrict y,
    const inc_t rs_y0,
    const inc_t cs_y0
    )
{
	// This is the panel dimension assumed by the packm kernel.
	const dim_t      mr    = 6;
	const dim_t      nr    = 8;

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.

	const uint64_t n = n0;

	const uint64_t m_full = m / 4;
	const uint64_t m_partial = (m % 4) / 2;
	const uint64_t m_left = (m % 4) % 2;

	const uint64_t rs_x = rs_x0;
	const uint64_t cs_x = cs_x0;

	const uint64_t rs_y = rs_y0;
	const uint64_t cs_y = cs_y0;

	// -------------------------------------------------------------------------

	if ( n > 0 && m_full > 0 || m_partial > 0 )
	{
		begin_asm()

		mov(var(x), rax)                   // load address of x.

		mov(var(m_full), r13)
		mov(var(m_partial), r14)

		mov(var(rs_x), r8)                 // load rs_x
		mov(var(cs_x), r10)                // load cs_x
		lea(mem(, r8, 8), r8)            // rs_x *= sizeof(double)
		lea(mem(, r10, 8), r10)            // cs_x *= sizeof(double)

		// mov(var(rs_y), r14)             should be 1
		mov(var(cs_y), r15)                // load cs_y
		lea(mem(, r15, 8), r15)            // cs_y *= sizeof(double)

		mov(var(y), rbx)                   // load address of y.

		// lea(mem(   , r10, 4), r14)         // r14 = 4*lda

		mov(var(alpha), rcx)               // load address of alpha

		vbroadcastsd(mem(rcx), ymm8)       // broadcast alpha

		// -- kappa unit, column storage on A --------------------------------------

		label(.DCOLUNIT)

		test(r13, r13)					   // check full left
		je(.DMFULLDONE)

		///
		mov(var(n), rsi)				   // i = n
		mov(var(x), rax)
		mov(var(y), rbx)

		label(.DMFULL)

		vmovupd(rax, ymm1)						   // load C block
		vmovupd(rbx, ymm2)						   // load buffer
		vfmadd132pd(ymm1, ymm2, ymm8)
		vmovupd(ymm1, mem(rax))
		
		add(r10, rax)
		add(r15, rbx)

		dec(rsi)
		jne(.DMFULL)                   // iterate again if i != 0.

		label(.DMFULLDONE)

		test(r14, r14)
		je(.DMPARTIALDONE)

		mov(var(n), rsi)
		mov(var(x), rax)
		mov(var(y), rbx)

		add(r8, rax)
		add(imm(4*8), rbx)

		label(.DMPARTIAL)

		vmovupd(rax, xmm1)						   // load C block
		vmovupd(rbx, xmm2)						   // load buffer
		vfmadd132pd(xmm1, xmm2, xmm8)
		vmovupd(xmm1, mem(rax))

		add(r8, rax)
		add(r15, rbx)

		dec(rsi)
		jne(.DMPARTIAL)

		label(.DMPARTIALDONE)

		end_asm(
		: // output operands (none)
		: // input operands
		  [m_full] "m" (m_full),
		  [m_partial] "m" (m_partial),
		  [x]      "m" (x),
		  [cs_x]   "m" (cs_x),
		  [y]      "m" (y),
		  [cs_y]   "m" (cs_y),
		  [alpha] "m"  (alpha),
		  [n]	   "m" (n),
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
		  "r8", /*"r9",*/ "r10", /*"r11",*/ "r12", "r13", "r14", "r15",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
		)
	}
	else {
		blis_daxpbys_mxn( m, n0,
		                  alpha, x, rs_x0, cs_x0,
		                  y, rs_y0, cs_y0 );
		return;
	}

	if (m_left > 0) {
		const uint64_t m_used = m_full * 4 + m_partial * 2;
		blis_daxpbys_mxn( m_left, n0,
		                  alpha,
		                  x + rs_x0 * m_used, rs_x0, cs_x0,
		                  y + rs_y0 * m_used, rs_y0, cs_y0
		                );
	}
}
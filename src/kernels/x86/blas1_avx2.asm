; blas-1 in avx2: axpy, dot, scal, f32 and f64
;
; axpy and scal are element wise so they stay bit exact with the c reference,
; plain vmulps/vaddps, no fma. dot is a reduction and goes the other way, four
; lane accumulators and fma for throughput, which reorders the sum so it does
; not match the sequential c dot to the bit. the harness checks dot against an
; f64 truth with an error bound instead of an exact compare
;
; sysv amd64:
;   axpy  y[rdi] x[rsi] n[rdx] a[xmm0]
;   dot   x[rdi] y[rsi] n[rdx] -> xmm0
;   scal  x[rdi] n[rsi] a[xmm0]

section .text

global stride_axpy_f32_avx2
global stride_axpy_f64_avx2
global stride_dot_f32_avx2
global stride_dot_f64_avx2
global stride_scal_f32_avx2
global stride_scal_f64_avx2

; ---- axpy ----

stride_axpy_f32_avx2:
    vbroadcastss ymm1, xmm0          ; a
    mov     rax, rdx
.b8:
    cmp     rax, 8
    jb      .tail
    vmovups ymm2, [rsi]              ; x
    vmulps  ymm2, ymm2, ymm1         ; a*x
    vmovups ymm3, [rdi]              ; y
    vaddps  ymm3, ymm3, ymm2         ; y + a*x
    vmovups [rdi], ymm3
    add     rsi, 32
    add     rdi, 32
    sub     rax, 8
    jmp     .b8
.tail:
    test    rax, rax
    jz      .done
.t1:
    vmovss  xmm2, [rsi]
    vmulss  xmm2, xmm2, xmm0
    vmovss  xmm3, [rdi]
    vaddss  xmm3, xmm3, xmm2
    vmovss  [rdi], xmm3
    add     rsi, 4
    add     rdi, 4
    dec     rax
    jnz     .t1
.done:
    vzeroupper
    ret

stride_axpy_f64_avx2:
    vbroadcastsd ymm1, xmm0
    mov     rax, rdx
.b4:
    cmp     rax, 4
    jb      .tail
    vmovupd ymm2, [rsi]
    vmulpd  ymm2, ymm2, ymm1
    vmovupd ymm3, [rdi]
    vaddpd  ymm3, ymm3, ymm2
    vmovupd [rdi], ymm3
    add     rsi, 32
    add     rdi, 32
    sub     rax, 4
    jmp     .b4
.tail:
    test    rax, rax
    jz      .done
.t1:
    vmovsd  xmm2, [rsi]
    vmulsd  xmm2, xmm2, xmm0
    vmovsd  xmm3, [rdi]
    vaddsd  xmm3, xmm3, xmm2
    vmovsd  [rdi], xmm3
    add     rsi, 8
    add     rdi, 8
    dec     rax
    jnz     .t1
.done:
    vzeroupper
    ret

; ---- scal ----

stride_scal_f32_avx2:
    vbroadcastss ymm1, xmm0          ; a
    mov     rax, rsi                 ; n
.b8:
    cmp     rax, 8
    jb      .tail
    vmovups ymm2, [rdi]
    vmulps  ymm2, ymm2, ymm1
    vmovups [rdi], ymm2
    add     rdi, 32
    sub     rax, 8
    jmp     .b8
.tail:
    test    rax, rax
    jz      .done
.t1:
    vmovss  xmm2, [rdi]
    vmulss  xmm2, xmm2, xmm0
    vmovss  [rdi], xmm2
    add     rdi, 4
    dec     rax
    jnz     .t1
.done:
    vzeroupper
    ret

stride_scal_f64_avx2:
    vbroadcastsd ymm1, xmm0
    mov     rax, rsi
.b4:
    cmp     rax, 4
    jb      .tail
    vmovupd ymm2, [rdi]
    vmulpd  ymm2, ymm2, ymm1
    vmovupd [rdi], ymm2
    add     rdi, 32
    sub     rax, 4
    jmp     .b4
.tail:
    test    rax, rax
    jz      .done
.t1:
    vmovsd  xmm2, [rdi]
    vmulsd  xmm2, xmm2, xmm0
    vmovsd  [rdi], xmm2
    add     rdi, 8
    dec     rax
    jnz     .t1
.done:
    vzeroupper
    ret

; ---- dot ----
; four accumulators to hide the fma latency, then horizontal reduce, then a
; scalar tail. the four way split is the reorder that breaks bit exactness

stride_dot_f32_avx2:
    vxorps  xmm0, xmm0, xmm0         ; clears the full ymm0..3
    vxorps  xmm1, xmm1, xmm1
    vxorps  xmm2, xmm2, xmm2
    vxorps  xmm3, xmm3, xmm3
    mov     rax, rdx
.b32:
    cmp     rax, 32
    jb      .b8
    vmovups ymm4, [rdi]
    vfmadd231ps ymm0, ymm4, [rsi]
    vmovups ymm5, [rdi + 32]
    vfmadd231ps ymm1, ymm5, [rsi + 32]
    vmovups ymm6, [rdi + 64]
    vfmadd231ps ymm2, ymm6, [rsi + 64]
    vmovups ymm7, [rdi + 96]
    vfmadd231ps ymm3, ymm7, [rsi + 96]
    add     rdi, 128
    add     rsi, 128
    sub     rax, 32
    jmp     .b32
.b8:
    cmp     rax, 8
    jb      .reduce
    vmovups ymm4, [rdi]
    vfmadd231ps ymm0, ymm4, [rsi]
    add     rdi, 32
    add     rsi, 32
    sub     rax, 8
    jmp     .b8
.reduce:
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2         ; 8 partials in ymm0
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1         ; 4 partials
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0         ; xmm0[0] = sum
.tail:
    test    rax, rax
    jz      .done
.t1:
    vmovss  xmm4, [rdi]
    vfmadd231ss xmm0, xmm4, [rsi]
    add     rdi, 4
    add     rsi, 4
    dec     rax
    jnz     .t1
.done:
    vzeroupper
    ret

stride_dot_f64_avx2:
    vxorpd  xmm0, xmm0, xmm0
    vxorpd  xmm1, xmm1, xmm1
    vxorpd  xmm2, xmm2, xmm2
    vxorpd  xmm3, xmm3, xmm3
    mov     rax, rdx
.b16:
    cmp     rax, 16
    jb      .b4
    vmovupd ymm4, [rdi]
    vfmadd231pd ymm0, ymm4, [rsi]
    vmovupd ymm5, [rdi + 32]
    vfmadd231pd ymm1, ymm5, [rsi + 32]
    vmovupd ymm6, [rdi + 64]
    vfmadd231pd ymm2, ymm6, [rsi + 64]
    vmovupd ymm7, [rdi + 96]
    vfmadd231pd ymm3, ymm7, [rsi + 96]
    add     rdi, 128
    add     rsi, 128
    sub     rax, 16
    jmp     .b16
.b4:
    cmp     rax, 4
    jb      .reduce
    vmovupd ymm4, [rdi]
    vfmadd231pd ymm0, ymm4, [rsi]
    add     rdi, 32
    add     rsi, 32
    sub     rax, 4
    jmp     .b4
.reduce:
    vaddpd  ymm0, ymm0, ymm1
    vaddpd  ymm2, ymm2, ymm3
    vaddpd  ymm0, ymm0, ymm2         ; 4 partials
    vextractf128 xmm1, ymm0, 1
    vaddpd  xmm0, xmm0, xmm1         ; 2 partials
    vhaddpd xmm0, xmm0, xmm0         ; xmm0[0] = sum
.tail:
    test    rax, rax
    jz      .done
.t1:
    vmovsd  xmm4, [rdi]
    vfmadd231sd xmm0, xmm4, [rsi]
    add     rdi, 8
    add     rsi, 8
    dec     rax
    jnz     .t1
.done:
    vzeroupper
    ret

section .note.GNU-stack noalloc noexec nowrite progbits

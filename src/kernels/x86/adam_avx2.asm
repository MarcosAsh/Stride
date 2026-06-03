; fused adam update, avx2, single streaming pass
;
; one pass over params m v grads, the whole adam step with the moments
; updated and written back in register. the naive version makes five passes
; over the arrays, this makes one
;
; bit exactness, the c reference compiled with std c11 emits no fma so the
; per element op sequence here uses plain vmulps vaddps vdivps vsqrtps in the
; same order as the scalar code, every ieee op is correctly rounded and the
; lanes are independent so the result matches scalar c to the last bit, the
; checkasm harness checks this at max ulp 0
;
; one ordering detail, c evaluates (1-beta2)*g*g left to right as
; ((1-beta2)*g)*g not (1-beta2)*(g*g) so the asm does the two multiplies in
; that order or the low bit drifts
;
; sysv amd64, args
;   rdi params  rsi m  rdx v  rcx grads  r8 n
;   xmm0 lr  xmm1 beta1  xmm2 beta2  xmm3 eps  xmm4 bc1  xmm5 bc2

section .text

global stride_adam_step_f32_avx2
global stride_adam_step_f64_avx2

; ---- f32, 8 lanes per vector iteration ----
stride_adam_step_f32_avx2:
    ; omb1 = 1 - beta1, omb2 = 1 - beta2, kept as scalars for the tail too
    mov     eax, 0x3f800000          ; 1.0f
    vmovd   xmm14, eax
    vsubss  xmm6, xmm14, xmm1        ; omb1
    vsubss  xmm7, xmm14, xmm2        ; omb2

    ; broadcast the loop invariant constants into ymm8..15
    vbroadcastss ymm8,  xmm1         ; beta1
    vbroadcastss ymm9,  xmm6         ; omb1
    vbroadcastss ymm10, xmm2         ; beta2
    vbroadcastss ymm11, xmm7         ; omb2
    vbroadcastss ymm12, xmm4         ; bc1
    vbroadcastss ymm13, xmm5         ; bc2
    vbroadcastss ymm14, xmm3         ; eps  (reuses the reg that held 1.0f)
    vbroadcastss ymm15, xmm0         ; lr

    mov     rax, r8                  ; remaining elements
.body8:
    cmp     rax, 8
    jb      .tail
    vmovups ymm0, [rcx]              ; g
    vmovups ymm1, [rsi]              ; m
    vmovups ymm2, [rdx]              ; v

    vmulps  ymm4, ymm1, ymm8         ; beta1*m
    vmulps  ymm5, ymm0, ymm9         ; omb1*g
    vaddps  ymm1, ymm4, ymm5         ; m_new
    vmovups [rsi], ymm1

    vmulps  ymm4, ymm2, ymm10        ; beta2*v
    vmulps  ymm5, ymm0, ymm11        ; omb2*g
    vmulps  ymm5, ymm5, ymm0         ; (omb2*g)*g
    vaddps  ymm2, ymm4, ymm5         ; v_new
    vmovups [rdx], ymm2

    vdivps  ymm1, ymm1, ymm12        ; m_hat = m_new/bc1
    vdivps  ymm2, ymm2, ymm13        ; v_hat = v_new/bc2
    vsqrtps ymm2, ymm2               ; sqrt(v_hat)
    vaddps  ymm2, ymm2, ymm14        ; denom = sqrt(v_hat)+eps
    vmulps  ymm1, ymm1, ymm15        ; lr*m_hat
    vdivps  ymm1, ymm1, ymm2         ; update

    vmovups ymm3, [rdi]
    vsubps  ymm3, ymm3, ymm1         ; params - update
    vmovups [rdi], ymm3

    add     rcx, 32
    add     rsi, 32
    add     rdx, 32
    add     rdi, 32
    sub     rax, 8
    jmp     .body8
.tail:
    test    rax, rax
    jz      .done
    ; the vector body clobbers xmm0..5 (low lanes of its working ymm regs),
    ; so the tail reads the constants from the ymm8..15 broadcasts instead and
    ; uses xmm0..3 as scratch
.tail1:
    vmovss  xmm0, [rcx]              ; g
    vmovss  xmm1, [rsi]
    vmulss  xmm1, xmm1, xmm8         ; beta1*m
    vmulss  xmm3, xmm0, xmm9         ; omb1*g
    vaddss  xmm1, xmm1, xmm3         ; m_new
    vmovss  [rsi], xmm1
    vmovss  xmm2, [rdx]
    vmulss  xmm2, xmm2, xmm10        ; beta2*v
    vmulss  xmm3, xmm0, xmm11        ; omb2*g
    vmulss  xmm3, xmm3, xmm0         ; (omb2*g)*g
    vaddss  xmm2, xmm2, xmm3         ; v_new
    vmovss  [rdx], xmm2
    vdivss  xmm1, xmm1, xmm12        ; m_hat
    vdivss  xmm2, xmm2, xmm13        ; v_hat
    vsqrtss xmm2, xmm2, xmm2
    vaddss  xmm2, xmm2, xmm14        ; denom
    vmulss  xmm1, xmm1, xmm15        ; lr*m_hat
    vdivss  xmm1, xmm1, xmm2         ; update
    vmovss  xmm3, [rdi]
    vsubss  xmm3, xmm3, xmm1
    vmovss  [rdi], xmm3
    add     rcx, 4
    add     rsi, 4
    add     rdx, 4
    add     rdi, 4
    dec     rax
    jnz     .tail1
.done:
    vzeroupper
    ret

; ---- f64, 4 lanes per vector iteration ----
stride_adam_step_f64_avx2:
    mov     rax, 0x3ff0000000000000  ; 1.0
    vmovq   xmm14, rax
    vsubsd  xmm6, xmm14, xmm1        ; omb1
    vsubsd  xmm7, xmm14, xmm2        ; omb2

    vbroadcastsd ymm8,  xmm1         ; beta1
    vbroadcastsd ymm9,  xmm6         ; omb1
    vbroadcastsd ymm10, xmm2         ; beta2
    vbroadcastsd ymm11, xmm7         ; omb2
    vbroadcastsd ymm12, xmm4         ; bc1
    vbroadcastsd ymm13, xmm5         ; bc2
    vbroadcastsd ymm14, xmm3         ; eps
    vbroadcastsd ymm15, xmm0         ; lr

    mov     rax, r8
.body4:
    cmp     rax, 4
    jb      .tail
    vmovupd ymm0, [rcx]              ; g
    vmovupd ymm1, [rsi]              ; m
    vmovupd ymm2, [rdx]              ; v

    vmulpd  ymm4, ymm1, ymm8
    vmulpd  ymm5, ymm0, ymm9
    vaddpd  ymm1, ymm4, ymm5         ; m_new
    vmovupd [rsi], ymm1

    vmulpd  ymm4, ymm2, ymm10
    vmulpd  ymm5, ymm0, ymm11
    vmulpd  ymm5, ymm5, ymm0         ; (omb2*g)*g
    vaddpd  ymm2, ymm4, ymm5         ; v_new
    vmovupd [rdx], ymm2

    vdivpd  ymm1, ymm1, ymm12        ; m_hat
    vdivpd  ymm2, ymm2, ymm13        ; v_hat
    vsqrtpd ymm2, ymm2
    vaddpd  ymm2, ymm2, ymm14        ; denom
    vmulpd  ymm1, ymm1, ymm15        ; lr*m_hat
    vdivpd  ymm1, ymm1, ymm2         ; update

    vmovupd ymm3, [rdi]
    vsubpd  ymm3, ymm3, ymm1
    vmovupd [rdi], ymm3

    add     rcx, 32
    add     rsi, 32
    add     rdx, 32
    add     rdi, 32
    sub     rax, 4
    jmp     .body4
.tail:
    test    rax, rax
    jz      .done
    ; constants from the ymm8..15 broadcasts, xmm0..3 as scratch (see f32)
.tail1:
    vmovsd  xmm0, [rcx]              ; g
    vmovsd  xmm1, [rsi]
    vmulsd  xmm1, xmm1, xmm8         ; beta1*m
    vmulsd  xmm3, xmm0, xmm9         ; omb1*g
    vaddsd  xmm1, xmm1, xmm3         ; m_new
    vmovsd  [rsi], xmm1
    vmovsd  xmm2, [rdx]
    vmulsd  xmm2, xmm2, xmm10        ; beta2*v
    vmulsd  xmm3, xmm0, xmm11        ; omb2*g
    vmulsd  xmm3, xmm3, xmm0         ; (omb2*g)*g
    vaddsd  xmm2, xmm2, xmm3         ; v_new
    vmovsd  [rdx], xmm2
    vdivsd  xmm1, xmm1, xmm12        ; m_hat
    vdivsd  xmm2, xmm2, xmm13        ; v_hat
    vsqrtsd xmm2, xmm2, xmm2
    vaddsd  xmm2, xmm2, xmm14        ; denom
    vmulsd  xmm1, xmm1, xmm15        ; lr*m_hat
    vdivsd  xmm1, xmm1, xmm2         ; update
    vmovsd  xmm3, [rdi]
    vsubsd  xmm3, xmm3, xmm1
    vmovsd  [rdi], xmm3
    add     rcx, 8
    add     rsi, 8
    add     rdx, 8
    add     rdi, 8
    dec     rax
    jnz     .tail1
.done:
    vzeroupper
    ret

; mark the stack non executable, otherwise nasm elf objects flag it
section .note.GNU-stack noalloc noexec nowrite progbits

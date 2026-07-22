
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx950
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z10reduce_sumPKDv4_fPdm ; -- Begin function _Z10reduce_sumPKDv4_fPdm
	.globl	_Z10reduce_sumPKDv4_fPdm
	.p2align	8
	.type	_Z10reduce_sumPKDv4_fPdm,@function
_Z10reduce_sumPKDv4_fPdm:               ; @_Z10reduce_sumPKDv4_fPdm
; %bb.0:
	s_load_dword s8, s[0:1], 0x24
	s_load_dwordx4 s[4:7], s[0:1], 0x8
	s_add_u32 s10, s0, 24
	s_addc_u32 s11, s1, 0
	v_mov_b32_e32 v1, 0
	s_waitcnt lgkmcnt(0)
	s_and_b32 s14, s8, 0xffff
	v_mov_b32_e32 v2, s2
	v_mad_u64_u32 v[2:3], s[8:9], s14, v2, v[0:1]
	s_mov_b32 s3, 0
	v_cmp_gt_u64_e32 vcc, s[6:7], v[2:3]
	v_mov_b64_e32 v[4:5], 0
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_4
; %bb.1:
	s_load_dword s15, s[10:11], 0x0
	s_load_dwordx2 s[12:13], s[0:1], 0x0
	v_mov_b64_e32 v[4:5], 0
	s_mov_b64 s[10:11], 0
	s_waitcnt lgkmcnt(0)
	s_mul_hi_u32 s1, s14, s15
	s_mul_i32 s0, s14, s15
	v_lshl_add_u64 v[6:7], v[2:3], 4, s[12:13]
	s_lshl_b64 s[12:13], s[0:1], 4
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	global_load_dwordx4 v[8:11], v[6:7], off
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[0:1]
	v_cmp_le_u64_e32 vcc, s[6:7], v[2:3]
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[12:13]
	s_or_b64 s[10:11], vcc, s[10:11]
	s_waitcnt vmcnt(0)
	v_cvt_f64_f32_e32 v[12:13], v8
	v_cvt_f64_f32_e32 v[8:9], v9
	v_cvt_f64_f32_e32 v[14:15], v10
	v_add_f64 v[8:9], v[12:13], v[8:9]
	v_cvt_f64_f32_e32 v[10:11], v11
	v_add_f64 v[8:9], v[8:9], v[14:15]
	v_add_f64 v[8:9], v[8:9], v[10:11]
	v_add_f64 v[4:5], v[4:5], v[8:9]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_2
; %bb.3:
	s_or_b64 exec, exec, s[10:11]
.LBB0_4:
	s_or_b64 exec, exec, s[8:9]
	v_lshlrev_b32_e32 v1, 3, v0
	s_cmp_lt_u32 s14, 2
	ds_write_b64 v1, v[4:5]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB0_9
; %bb.5:
	s_lshr_b32 s6, s14, 1
	s_branch .LBB0_7
.LBB0_6:                                ;   in Loop: Header=BB0_7 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_lshr_b32 s0, s6, 1
	s_cmp_lt_u32 s6, 2
	s_mov_b32 s6, s0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB0_9
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	v_cmp_gt_u32_e32 vcc, s6, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.8:                                ;   in Loop: Header=BB0_7 Depth=1
	v_lshl_add_u32 v2, s6, 3, v1
	ds_read_b64 v[2:3], v2
	ds_read_b64 v[4:5], v1
	s_waitcnt lgkmcnt(0)
	v_add_f64 v[2:3], v[2:3], v[4:5]
	ds_write_b64 v1, v[2:3]
	s_branch .LBB0_6
.LBB0_9:
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_11
; %bb.10:
	v_mov_b32_e32 v2, 0
	ds_read_b64 v[0:1], v2
	s_lshl_b64 s[0:1], s[2:3], 3
	s_add_u32 s0, s4, s0
	s_addc_u32 s1, s5, s1
	s_waitcnt lgkmcnt(0)
	global_store_dwordx2 v2, v[0:1], s[0:1]
.LBB0_11:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z10reduce_sumPKDv4_fPdm
		.amdhsa_group_segment_fixed_size 2048
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 16
		.amdhsa_next_free_sgpr 16
		.amdhsa_accum_offset 16
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z10reduce_sumPKDv4_fPdm, .Lfunc_end0-_Z10reduce_sumPKDv4_fPdm
                                        ; -- End function
	.set _Z10reduce_sumPKDv4_fPdm.num_vgpr, 16
	.set _Z10reduce_sumPKDv4_fPdm.num_agpr, 0
	.set _Z10reduce_sumPKDv4_fPdm.numbered_sgpr, 16
	.set _Z10reduce_sumPKDv4_fPdm.num_named_barrier, 0
	.set _Z10reduce_sumPKDv4_fPdm.private_seg_size, 0
	.set _Z10reduce_sumPKDv4_fPdm.uses_vcc, 1
	.set _Z10reduce_sumPKDv4_fPdm.uses_flat_scratch, 0
	.set _Z10reduce_sumPKDv4_fPdm.has_dyn_sized_stack, 0
	.set _Z10reduce_sumPKDv4_fPdm.has_recursion, 0
	.set _Z10reduce_sumPKDv4_fPdm.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 396
; TotalNumSgprs: 22
; NumVgprs: 16
; NumAgprs: 0
; TotalNumVgprs: 16
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 2048 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 16
; AccumOffset: 16
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 3
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z11fill_kernelPDv4_fm  ; -- Begin function _Z11fill_kernelPDv4_fm
	.globl	_Z11fill_kernelPDv4_fm
	.p2align	8
	.type	_Z11fill_kernelPDv4_fm,@function
_Z11fill_kernelPDv4_fm:                 ; @_Z11fill_kernelPDv4_fm
; %bb.0:
	s_load_dword s3, s[0:1], 0x1c
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	s_add_u32 s0, s0, 16
	s_addc_u32 s1, s1, 0
	v_mov_b32_e32 v1, 0
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 0xffff
	v_mov_b32_e32 v2, s2
	v_mad_u64_u32 v[4:5], s[8:9], s3, v2, v[0:1]
	v_cmp_gt_u64_e32 vcc, s[6:7], v[4:5]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB1_3
; %bb.1:
	s_load_dword s0, s[0:1], 0x0
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	v_lshl_add_u64 v[6:7], v[4:5], 4, v[0:1]
	v_mov_b32_e32 v0, 1.0
	s_waitcnt lgkmcnt(0)
	s_mul_hi_u32 s1, s3, s0
	s_mul_i32 s0, s3, s0
	s_lshl_b64 s[2:3], s[0:1], 4
	s_mov_b64 s[4:5], 0
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
.LBB1_2:                                ; =>This Inner Loop Header: Depth=1
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[0:1]
	v_cmp_le_u64_e32 vcc, s[6:7], v[4:5]
	global_store_dwordx4 v[6:7], v[0:3], off
	s_or_b64 s[4:5], vcc, s[4:5]
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[2:3]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB1_2
.LBB1_3:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z11fill_kernelPDv4_fm
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 10
		.amdhsa_accum_offset 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end1:
	.size	_Z11fill_kernelPDv4_fm, .Lfunc_end1-_Z11fill_kernelPDv4_fm
                                        ; -- End function
	.set _Z11fill_kernelPDv4_fm.num_vgpr, 8
	.set _Z11fill_kernelPDv4_fm.num_agpr, 0
	.set _Z11fill_kernelPDv4_fm.numbered_sgpr, 10
	.set _Z11fill_kernelPDv4_fm.num_named_barrier, 0
	.set _Z11fill_kernelPDv4_fm.private_seg_size, 0
	.set _Z11fill_kernelPDv4_fm.uses_vcc, 1
	.set _Z11fill_kernelPDv4_fm.uses_flat_scratch, 0
	.set _Z11fill_kernelPDv4_fm.has_dyn_sized_stack, 0
	.set _Z11fill_kernelPDv4_fm.has_recursion, 0
	.set _Z11fill_kernelPDv4_fm.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 168
; TotalNumSgprs: 16
; NumVgprs: 8
; NumAgprs: 0
; TotalNumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 8
; AccumOffset: 8
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 1
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z17bandwidth_memreadILi8EEvPKDv4_fPfm,"axG",@progbits,_Z17bandwidth_memreadILi8EEvPKDv4_fPfm,comdat
	.protected	_Z17bandwidth_memreadILi8EEvPKDv4_fPfm ; -- Begin function _Z17bandwidth_memreadILi8EEvPKDv4_fPfm
	.globl	_Z17bandwidth_memreadILi8EEvPKDv4_fPfm
	.p2align	8
	.type	_Z17bandwidth_memreadILi8EEvPKDv4_fPfm,@function
_Z17bandwidth_memreadILi8EEvPKDv4_fPfm: ; @_Z17bandwidth_memreadILi8EEvPKDv4_fPfm
; %bb.0:
	s_load_dword s3, s[0:1], 0x24
	s_load_dword s36, s[0:1], 0x18
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	v_mov_b32_e32 v1, 0
	s_waitcnt lgkmcnt(0)
	s_and_b32 s33, s3, 0xffff
	v_mov_b32_e32 v2, s2
	v_mad_u64_u32 v[6:7], s[0:1], s33, v2, v[0:1]
	s_mul_hi_u32 s1, s33, s36
	s_mul_i32 s0, s33, s36
	s_lshl_b64 s[10:11], s[0:1], 3
	s_add_u32 s12, s8, -1
	s_addc_u32 s13, s9, -1
	v_lshl_add_u64 v[2:3], v[6:7], 0, s[10:11]
	v_cmp_ge_u64_e32 vcc, s[12:13], v[2:3]
	v_mov_b32_e32 v2, v1
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v4, v1
	v_mov_b32_e32 v5, v1
	v_mov_b64_e32 v[8:9], v[6:7]
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB2_4
; %bb.1:
	s_mul_i32 s3, s1, 48
	s_mul_hi_u32 s24, s0, 48
	s_add_i32 s25, s24, s3
	s_mul_i32 s3, s1, 0x50
	s_mul_hi_u32 s28, s0, 0x50
	s_add_i32 s29, s28, s3
	s_mul_i32 s3, s1, 0x60
	s_mul_hi_u32 s30, s0, 0x60
	s_mov_b32 s37, 0
	s_add_i32 s31, s30, s3
	s_mul_i32 s3, s1, 0x70
	s_mul_hi_u32 s34, s0, 0x70
	s_lshl_b64 s[18:19], s[0:1], 4
	s_lshl_b64 s[20:21], s[0:1], 7
	s_lshl_b64 s[22:23], s[0:1], 5
	s_lshl_b64 s[26:27], s[0:1], 6
	s_add_i32 s35, s34, s3
	s_lshl_b64 s[36:37], s[36:37], 3
	s_mul_hi_u32 s17, s33, s2
	s_mul_i32 s16, s33, s2
	s_add_u32 s2, s36, s2
	s_addc_u32 s3, s37, 0
	s_mul_i32 s3, s3, s33
	s_mul_hi_u32 s36, s2, s33
	v_lshl_add_u64 v[4:5], v[6:7], 4, s[4:5]
	s_mul_i32 s24, s0, 48
	s_mul_i32 s28, s0, 0x50
	s_mul_i32 s30, s0, 0x60
	s_mul_i32 s34, s0, 0x70
	s_add_i32 s3, s36, s3
	s_mul_i32 s2, s2, s33
	s_mov_b64 s[36:37], 0
	v_mov_b64_e32 v[8:9], v[0:1]
	v_mov_b32_e32 v0, v1
	v_mov_b32_e32 v2, v1
	v_mov_b32_e32 v3, v1
.LBB2_2:                                ; =>This Inner Loop Header: Depth=1
	global_load_dwordx4 v[10:13], v[4:5], off nt
	v_lshl_add_u64 v[42:43], v[4:5], 0, s[18:19]
	v_lshl_add_u64 v[44:45], v[4:5], 0, s[22:23]
	v_lshl_add_u64 v[46:47], v[4:5], 0, s[24:25]
	v_lshl_add_u64 v[48:49], v[4:5], 0, s[26:27]
	v_lshl_add_u64 v[50:51], v[4:5], 0, s[28:29]
	v_lshl_add_u64 v[52:53], v[4:5], 0, s[30:31]
	v_lshl_add_u64 v[54:55], v[4:5], 0, s[34:35]
	global_load_dwordx4 v[14:17], v[42:43], off nt
	global_load_dwordx4 v[18:21], v[44:45], off nt
	global_load_dwordx4 v[22:25], v[46:47], off nt
	global_load_dwordx4 v[26:29], v[48:49], off nt
	global_load_dwordx4 v[30:33], v[50:51], off nt
	global_load_dwordx4 v[34:37], v[52:53], off nt
	global_load_dwordx4 v[38:41], v[54:55], off nt
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[10:11]
	v_lshl_add_u64 v[42:43], s[2:3], 0, v[8:9]
	v_cmp_lt_u64_e32 vcc, s[12:13], v[42:43]
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[20:21]
	s_or_b64 s[36:37], vcc, s[36:37]
	s_waitcnt vmcnt(7)
	v_pk_add_f32 v[2:3], v[2:3], v[12:13]
	v_pk_add_f32 v[0:1], v[0:1], v[10:11]
	s_waitcnt vmcnt(6)
	v_pk_add_f32 v[2:3], v[2:3], v[16:17]
	v_pk_add_f32 v[0:1], v[0:1], v[14:15]
	s_waitcnt vmcnt(5)
	v_pk_add_f32 v[2:3], v[2:3], v[20:21]
	v_pk_add_f32 v[0:1], v[0:1], v[18:19]
	s_waitcnt vmcnt(4)
	v_pk_add_f32 v[2:3], v[2:3], v[24:25]
	v_pk_add_f32 v[0:1], v[0:1], v[22:23]
	s_waitcnt vmcnt(3)
	v_pk_add_f32 v[2:3], v[2:3], v[28:29]
	v_pk_add_f32 v[0:1], v[0:1], v[26:27]
	s_waitcnt vmcnt(2)
	v_pk_add_f32 v[2:3], v[2:3], v[32:33]
	v_pk_add_f32 v[0:1], v[0:1], v[30:31]
	s_waitcnt vmcnt(1)
	v_pk_add_f32 v[2:3], v[2:3], v[36:37]
	v_pk_add_f32 v[0:1], v[0:1], v[34:35]
	s_waitcnt vmcnt(0)
	v_pk_add_f32 v[2:3], v[2:3], v[40:41]
	v_pk_add_f32 v[0:1], v[0:1], v[38:39]
	s_andn2_b64 exec, exec, s[36:37]
	s_cbranch_execnz .LBB2_2
; %bb.3:
	s_or_b64 exec, exec, s[36:37]
	v_mov_b64_e32 v[4:5], v[2:3]
	v_lshl_add_u64 v[8:9], s[16:17], 0, v[8:9]
	v_mov_b64_e32 v[2:3], v[0:1]
.LBB2_4:
	s_or_b64 exec, exec, s[14:15]
	v_cmp_gt_u64_e32 vcc, s[8:9], v[8:9]
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_8
; %bb.5:
	v_lshl_add_u64 v[0:1], v[8:9], 4, s[4:5]
	s_lshl_b64 s[10:11], s[0:1], 4
	s_mov_b64 s[4:5], 0
.LBB2_6:                                ; =>This Inner Loop Header: Depth=1
	global_load_dwordx4 v[10:13], v[0:1], off nt
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[0:1]
	v_cmp_le_u64_e32 vcc, s[8:9], v[8:9]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[10:11]
	s_or_b64 s[4:5], vcc, s[4:5]
	s_waitcnt vmcnt(0)
	v_pk_add_f32 v[4:5], v[4:5], v[12:13]
	v_pk_add_f32 v[2:3], v[2:3], v[10:11]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB2_6
; %bb.7:
	s_or_b64 exec, exec, s[4:5]
.LBB2_8:
	s_or_b64 exec, exec, s[2:3]
	v_cmp_eq_f32_e32 vcc, -1.0, v2
	v_cmp_eq_f32_e64 s[0:1], -2.0, v3
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB2_10
; %bb.9:
	v_add_f32_e32 v2, 0xc0400000, v4
	v_lshl_add_u64 v[0:1], v[6:7], 2, s[6:7]
	v_add_f32_e32 v2, v5, v2
	global_store_dword v[0:1], v2, off
.LBB2_10:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z17bandwidth_memreadILi8EEvPKDv4_fPfm
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 56
		.amdhsa_next_free_sgpr 38
		.amdhsa_accum_offset 56
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z17bandwidth_memreadILi8EEvPKDv4_fPfm,"axG",@progbits,_Z17bandwidth_memreadILi8EEvPKDv4_fPfm,comdat
.Lfunc_end2:
	.size	_Z17bandwidth_memreadILi8EEvPKDv4_fPfm, .Lfunc_end2-_Z17bandwidth_memreadILi8EEvPKDv4_fPfm
                                        ; -- End function
	.set _Z17bandwidth_memreadILi8EEvPKDv4_fPfm.num_vgpr, 56
	.set _Z17bandwidth_memreadILi8EEvPKDv4_fPfm.num_agpr, 0
	.set _Z17bandwidth_memreadILi8EEvPKDv4_fPfm.numbered_sgpr, 38
	.set _Z17bandwidth_memreadILi8EEvPKDv4_fPfm.num_named_barrier, 0
	.set _Z17bandwidth_memreadILi8EEvPKDv4_fPfm.private_seg_size, 0
	.set _Z17bandwidth_memreadILi8EEvPKDv4_fPfm.uses_vcc, 1
	.set _Z17bandwidth_memreadILi8EEvPKDv4_fPfm.uses_flat_scratch, 0
	.set _Z17bandwidth_memreadILi8EEvPKDv4_fPfm.has_dyn_sized_stack, 0
	.set _Z17bandwidth_memreadILi8EEvPKDv4_fPfm.has_recursion, 0
	.set _Z17bandwidth_memreadILi8EEvPKDv4_fPfm.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 800
; TotalNumSgprs: 44
; NumVgprs: 56
; NumAgprs: 0
; TotalNumVgprs: 56
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 6
; NumSGPRsForWavesPerEU: 44
; NumVGPRsForWavesPerEU: 56
; AccumOffset: 56
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 13
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_7037446ac5eaa0ac,@object ; @__hip_cuid_7037446ac5eaa0ac
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_7037446ac5eaa0ac
__hip_cuid_7037446ac5eaa0ac:
	.byte	0                               ; 0x0
	.size	__hip_cuid_7037446ac5eaa0ac, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_7037446ac5eaa0ac
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           8
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 2048
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z10reduce_sumPKDv4_fPdm
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         _Z10reduce_sumPKDv4_fPdm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     16
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           8
        .value_kind:     by_value
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z11fill_kernelPDv4_fm
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         _Z11fill_kernelPDv4_fm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     8
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           8
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z17bandwidth_memreadILi8EEvPKDv4_fPfm
    .private_segment_fixed_size: 0
    .sgpr_count:     44
    .sgpr_spill_count: 0
    .symbol:         _Z17bandwidth_memreadILi8EEvPKDv4_fPfm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     56
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx950

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu-
	.file	"bandwidth_memread.hip"
	.text
	.globl	_Z25__device_stub__reduce_sumPKDv4_fPdm # -- Begin function _Z25__device_stub__reduce_sumPKDv4_fPdm
	.p2align	4
	.type	_Z25__device_stub__reduce_sumPKDv4_fPdm,@function
_Z25__device_stub__reduce_sumPKDv4_fPdm: # @_Z25__device_stub__reduce_sumPKDv4_fPdm
	.cfi_startproc
# %bb.0:
	subq	$104, %rsp
	.cfi_def_cfa_offset 112
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdx, 56(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$_Z10reduce_sumPKDv4_fPdm, %edi
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end0:
	.size	_Z25__device_stub__reduce_sumPKDv4_fPdm, .Lfunc_end0-_Z25__device_stub__reduce_sumPKDv4_fPdm
	.cfi_endproc
                                        # -- End function
	.globl	_Z26__device_stub__fill_kernelPDv4_fm # -- Begin function _Z26__device_stub__fill_kernelPDv4_fm
	.p2align	4
	.type	_Z26__device_stub__fill_kernelPDv4_fm,@function
_Z26__device_stub__fill_kernelPDv4_fm:  # @_Z26__device_stub__fill_kernelPDv4_fm
	.cfi_startproc
# %bb.0:
	subq	$88, %rsp
	.cfi_def_cfa_offset 96
	movq	%rdi, 56(%rsp)
	movq	%rsi, 48(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	64(%rsp), %r9
	movl	$_Z11fill_kernelPDv4_fm, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$104, %rsp
	.cfi_adjust_cfa_offset -104
	retq
.Lfunc_end1:
	.size	_Z26__device_stub__fill_kernelPDv4_fm, .Lfunc_end1-_Z26__device_stub__fill_kernelPDv4_fm
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function main
.LCPI2_0:
	.quad	0x408f400000000000              # double 1000
.LCPI2_1:
	.quad	0xc170000000000000              # double -16777216
.LCPI2_3:
	.quad	0x3e70000000000000              # double 5.9604644775390625E-8
.LCPI2_4:
	.quad	0x3e112e0be826d695              # double 1.0000000000000001E-9
.LCPI2_5:
	.quad	0x4170000000000000              # double 16777216
.LCPI2_8:
	.quad	0x4049000000000000              # double 50
.LCPI2_9:
	.quad	0x3f50624dd2f1a9fc              # double 0.001
.LCPI2_10:
	.quad	0x41cdcd6500000000              # double 1.0E+9
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0
.LCPI2_2:
	.quad	0x7fffffffffffffff              # double NaN
	.quad	0x7fffffffffffffff              # double NaN
.LCPI2_6:
	.long	1127219200                      # 0x43300000
	.long	1160773632                      # 0x45300000
	.long	0                               # 0x0
	.long	0                               # 0x0
.LCPI2_7:
	.quad	0x4330000000000000              # double 4503599627370496
	.quad	0x4530000000000000              # double 1.9342813113834067E+25
	.text
	.globl	main
	.p2align	4
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception0
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$1624, %rsp                     # imm = 0x658
	.cfi_def_cfa_offset 1680
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	.cfi_escape 0x2e, 0x00
	leaq	152(%rsp), %rdi
	xorl	%esi, %esi
	callq	hipGetDevicePropertiesR0600
	testl	%eax, %eax
	jne	.LBB2_1
# %bb.2:
	leaq	1312(%rsp), %rdx
	movl	540(%rsp), %ecx
	cvtsi2sdl	500(%rsp), %xmm0
	divsd	.LCPI2_0(%rip), %xmm0
	.cfi_escape 0x2e, 0x00
	leaq	152(%rsp), %rsi
	movl	$.L.str.2, %edi
	movb	$1, %al
	callq	printf
	movslq	540(%rsp), %rbx
	movq	$0, (%rsp)
	movq	$0, 8(%rsp)
	.cfi_escape 0x2e, 0x00
	movq	%rsp, %rdi
	movl	$67108864, %esi                 # imm = 0x4000000
	callq	hipMalloc
	testl	%eax, %eax
	jne	.LBB2_3
# %bb.4:
	leaq	(%rbx,%rbx), %r14
	leaq	(,%r14,8), %rbp
	.cfi_escape 0x2e, 0x00
	leaq	8(%rsp), %rdi
	movq	%rbp, %rsi
	callq	hipMalloc
	testl	%eax, %eax
	jne	.LBB2_5
# %bb.6:
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	%r14d, %eax
	leaq	(%rax,%rdx), %r15
	addq	$-256, %r15
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB2_8
# %bb.7:
	movq	(%rsp), %rax
	movq	%rax, 72(%rsp)
	movq	$4194304, 64(%rsp)              # imm = 0x400000
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	56(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	leaq	80(%rsp), %r9
	movl	$_Z11fill_kernelPDv4_fm, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	64(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB2_8:
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError
	testl	%eax, %eax
	jne	.LBB2_9
# %bb.10:
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movl	$1, %esi
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB2_12
# %bb.11:
	movq	(%rsp), %rax
	movq	8(%rsp), %rcx
	movq	%rax, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	$4194304, 56(%rsp)              # imm = 0x400000
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	104(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	leaq	80(%rsp), %r9
	movl	$_Z10reduce_sumPKDv4_fPdm, %edi
	pushq	104(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB2_12:
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError
	testl	%eax, %eax
	jne	.LBB2_13
# %bb.14:
	testl	%ebx, %ebx
	js	.LBB2_84
# %bb.15:
	je	.LBB2_16
# %bb.17:
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	callq	_Znwm
	movq	%rax, %r12
	movq	%r14, %r13
	leaq	(%rax,%r14,8), %rbx
	movq	$0, (%rax)
	leaq	8(%rax), %rdi
	leaq	-8(%rbp), %rdx
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%r12, %r14
	addq	%rbp, %r14
	jmp	.LBB2_18
.LBB2_16:
	movq	%r14, %r13
	xorl	%ebx, %ebx
	xorl	%r12d, %r12d
	xorl	%r14d, %r14d
.LBB2_18:
	movq	8(%rsp), %rsi
.Ltmp0:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movq	%rbp, %rdx
	movl	$2, %ecx
	callq	hipMemcpy
.Ltmp1:                                 # EH_LABEL
# %bb.19:
	testl	%eax, %eax
	jne	.LBB2_34
# %bb.20:
	xorpd	%xmm0, %xmm0
	cmpq	%r14, %r12
	je	.LBB2_23
# %bb.21:
	movq	%r12, %rax
	.p2align	4
.LBB2_22:                               # =>This Inner Loop Header: Depth=1
	addsd	(%rax), %xmm0
	addq	$8, %rax
	cmpq	%r14, %rax
	jne	.LBB2_22
.LBB2_23:
	movsd	.LCPI2_1(%rip), %xmm1           # xmm1 = [-1.6777216E+7,0.0E+0]
	addsd	%xmm0, %xmm1
	andpd	.LCPI2_2(%rip), %xmm1
	mulsd	.LCPI2_3(%rip), %xmm1
	movsd	.LCPI2_4(%rip), %xmm2           # xmm2 = [1.0000000000000001E-9,0.0E+0]
	ucomisd	%xmm1, %xmm2
	movapd	%xmm1, %xmm2
	movapd	%xmm1, 128(%rsp)                # 16-byte Spill
	movl	$.L.str.4, %eax
	movl	$.L.str.5, %esi
	cmovaq	%rax, %rsi
	.cfi_escape 0x2e, 0x00
	movsd	.LCPI2_5(%rip), %xmm1           # xmm1 = [1.6777216E+7,0.0E+0]
	movl	$.L.str.3, %edi
	movb	$3, %al
	callq	printf
	movsd	.LCPI2_4(%rip), %xmm0           # xmm0 = [1.0000000000000001E-9,0.0E+0]
	ucomisd	128(%rsp), %xmm0                # 16-byte Folded Reload
	jbe	.LBB2_24
# %bb.37:
	movq	(%rsp), %rdi
.Ltmp5:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree
.Ltmp6:                                 # EH_LABEL
# %bb.38:
	testl	%eax, %eax
	jne	.LBB2_39
# %bb.42:
	movq	%r13, %r14
	movq	8(%rsp), %rdi
.Ltmp10:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree
.Ltmp11:                                # EH_LABEL
# %bb.43:
	xorl	%ebp, %ebp
	testl	%eax, %eax
	jne	.LBB2_44
# %bb.25:
	testq	%r12, %r12
	je	.LBB2_27
.LBB2_26:
	subq	%r12, %rbx
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movq	%rbx, %rsi
	callq	_ZdlPvm
.LBB2_27:
	movsd	.LCPI2_4(%rip), %xmm0           # xmm0 = [1.0000000000000001E-9,0.0E+0]
	ucomisd	128(%rsp), %xmm0                # 16-byte Folded Reload
	jbe	.LBB2_33
# %bb.28:
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.7, %edi
	movl	$.L.str.8, %esi
	movl	$.L.str.9, %edx
	movl	$.L.str.10, %ecx
	xorl	%eax, %eax
	callq	printf
	shlq	$10, %r14
	movq	%rsp, %r13
	xorl	%r12d, %r12d
	jmp	.LBB2_29
	.p2align	4
.LBB2_30:                               #   in Loop: Header=BB2_29 Depth=1
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.11, %edi
	movq	%rbp, %rsi
	xorl	%eax, %eax
	callq	printf
.LBB2_31:                               #   in Loop: Header=BB2_29 Depth=1
	addq	$8, %r12
	cmpq	$32, %r12
	je	.LBB2_32
.LBB2_29:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_67 Depth 2
	movq	.L__const.main.sizes_mib(%r12), %rbp
	movq	%rbp, %rbx
	shlq	$20, %rbx
	movq	$0, (%rsp)
	movq	$0, 8(%rsp)
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%rbx, %rsi
	callq	hipMalloc
	testl	%eax, %eax
	jne	.LBB2_30
# %bb.51:                               #   in Loop: Header=BB2_29 Depth=1
	.cfi_escape 0x2e, 0x00
	leaq	8(%rsp), %rdi
	movq	%r14, %rsi
	callq	hipMalloc
	testl	%eax, %eax
	jne	.LBB2_52
# %bb.53:                               #   in Loop: Header=BB2_29 Depth=1
	movq	%rbx, %r13
	shrq	$4, %r13
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movl	$1, %esi
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB2_55
# %bb.54:                               #   in Loop: Header=BB2_29 Depth=1
	movq	(%rsp), %rax
	movq	%rax, 72(%rsp)
	movq	%r13, 64(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	56(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movl	$_Z11fill_kernelPDv4_fm, %edi
	leaq	80(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	64(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB2_55:                               #   in Loop: Header=BB2_29 Depth=1
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB2_56
# %bb.57:                               #   in Loop: Header=BB2_29 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movl	$1, %esi
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB2_59
# %bb.58:                               #   in Loop: Header=BB2_29 Depth=1
	movq	(%rsp), %rax
	movq	8(%rsp), %rcx
	movq	%rax, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r13, 56(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	104(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movl	$_Z17bandwidth_memreadILi8EEvPKDv4_fPfm, %edi
	leaq	80(%rsp), %r9
	pushq	104(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB2_59:                               #   in Loop: Header=BB2_29 Depth=1
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB2_60
# %bb.61:                               #   in Loop: Header=BB2_29 Depth=1
	.cfi_escape 0x2e, 0x00
	leaq	120(%rsp), %rdi
	callq	hipEventCreate
	testl	%eax, %eax
	jne	.LBB2_62
# %bb.63:                               #   in Loop: Header=BB2_29 Depth=1
	movq	%rbp, 128(%rsp)                 # 8-byte Spill
	.cfi_escape 0x2e, 0x00
	leaq	112(%rsp), %rdi
	callq	hipEventCreate
	testl	%eax, %eax
	jne	.LBB2_64
# %bb.65:                               #   in Loop: Header=BB2_29 Depth=1
	movq	%r14, %rbp
	movq	120(%rsp), %rdi
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord
	testl	%eax, %eax
	jne	.LBB2_85
# %bb.66:                               #   in Loop: Header=BB2_29 Depth=1
	movl	$50, %r14d
	jmp	.LBB2_67
	.p2align	4
.LBB2_69:                               #   in Loop: Header=BB2_67 Depth=2
	decl	%r14d
	je	.LBB2_70
.LBB2_67:                               #   Parent Loop BB2_29 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movl	$1, %esi
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB2_69
# %bb.68:                               #   in Loop: Header=BB2_67 Depth=2
	movq	(%rsp), %rax
	movq	8(%rsp), %rcx
	movq	%rax, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r13, 56(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	104(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movl	$_Z17bandwidth_memreadILi8EEvPKDv4_fPfm, %edi
	leaq	80(%rsp), %r9
	pushq	104(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	jmp	.LBB2_69
	.p2align	4
.LBB2_70:                               #   in Loop: Header=BB2_29 Depth=1
	movq	112(%rsp), %rdi
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord
	testl	%eax, %eax
	jne	.LBB2_71
# %bb.72:                               #   in Loop: Header=BB2_29 Depth=1
	movq	112(%rsp), %rdi
	.cfi_escape 0x2e, 0x00
	callq	hipEventSynchronize
	testl	%eax, %eax
	jne	.LBB2_73
# %bb.74:                               #   in Loop: Header=BB2_29 Depth=1
	movq	%rbp, %r14
	movl	$0, 80(%rsp)
	movq	120(%rsp), %rsi
	movq	112(%rsp), %rdx
	.cfi_escape 0x2e, 0x00
	leaq	80(%rsp), %rdi
	callq	hipEventElapsedTime
	testl	%eax, %eax
	movq	%rsp, %r13
	movq	128(%rsp), %rsi                 # 8-byte Reload
	jne	.LBB2_75
# %bb.76:                               #   in Loop: Header=BB2_29 Depth=1
	movq	%rbx, %xmm1
	punpckldq	.LCPI2_6(%rip), %xmm1   # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	subpd	.LCPI2_7(%rip), %xmm1
	movapd	%xmm1, %xmm0
	unpckhpd	%xmm1, %xmm0                    # xmm0 = xmm0[1],xmm1[1]
	addsd	%xmm1, %xmm0
	movss	80(%rsp), %xmm1                 # xmm1 = mem[0],zero,zero,zero
	cvtss2sd	%xmm1, %xmm1
	mulsd	.LCPI2_8(%rip), %xmm0
	mulsd	.LCPI2_9(%rip), %xmm1
	divsd	%xmm1, %xmm0
	divsd	.LCPI2_10(%rip), %xmm0
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.12, %edi
	movl	$50, %edx
	movb	$1, %al
	callq	printf
	movq	120(%rsp), %rdi
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy
	testl	%eax, %eax
	jne	.LBB2_77
# %bb.78:                               #   in Loop: Header=BB2_29 Depth=1
	movq	112(%rsp), %rdi
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy
	testl	%eax, %eax
	jne	.LBB2_79
# %bb.80:                               #   in Loop: Header=BB2_29 Depth=1
	movq	(%rsp), %rdi
	.cfi_escape 0x2e, 0x00
	callq	hipFree
	testl	%eax, %eax
	jne	.LBB2_81
# %bb.82:                               #   in Loop: Header=BB2_29 Depth=1
	movq	8(%rsp), %rdi
	.cfi_escape 0x2e, 0x00
	callq	hipFree
	testl	%eax, %eax
	je	.LBB2_31
# %bb.83:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$191, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_24:
	.cfi_escape 0x2e, 0x00
	movl	$.Lstr, %edi
	callq	puts@PLT
	movl	$1, %ebp
	movq	%r13, %r14
	testq	%r12, %r12
	jne	.LBB2_26
	jmp	.LBB2_27
.LBB2_32:
	.cfi_escape 0x2e, 0x00
	movl	$.Lstr.15, %edi
	callq	puts@PLT
	xorl	%ebp, %ebp
.LBB2_33:
	movl	%ebp, %eax
	addq	$1624, %rsp                     # imm = 0x658
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB2_81:
	.cfi_def_cfa_offset 1680
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$190, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_64:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$176, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_75:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$184, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_56:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$167, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_85:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$177, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_77:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$188, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_60:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$171, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_71:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$180, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_79:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$189, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_62:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$175, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_73:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$181, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_52:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$165, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_1:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$115, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_3:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$128, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_5:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$129, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_9:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$132, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_13:
	movq	stderr(%rip), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%rbx, %rdi
	movq	%rax, %rdx
	movl	$134, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_84:
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.14, %edi
	callq	_ZSt20__throw_length_errorPKc
.LBB2_34:
	movq	stderr(%rip), %r14
.Ltmp2:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
.Ltmp3:                                 # EH_LABEL
# %bb.35:
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%r14, %rdi
	movq	%rax, %rdx
	movl	$138, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_39:
	movq	stderr(%rip), %r14
.Ltmp7:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
.Ltmp8:                                 # EH_LABEL
# %bb.40:
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%r14, %rdi
	movq	%rax, %rdx
	movl	$148, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_44:
	movq	stderr(%rip), %r14
.Ltmp12:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString
.Ltmp13:                                # EH_LABEL
# %bb.45:
	.cfi_escape 0x2e, 0x00
	movl	$.L.str, %esi
	movl	$.L.str.1, %ecx
	movq	%r14, %rdi
	movq	%rax, %rdx
	movl	$149, %r8d
	xorl	%eax, %eax
	callq	fprintf
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit
.LBB2_46:
.Ltmp14:                                # EH_LABEL
	jmp	.LBB2_48
.LBB2_41:
.Ltmp9:                                 # EH_LABEL
	jmp	.LBB2_48
.LBB2_47:
.Ltmp4:                                 # EH_LABEL
.LBB2_48:
	movq	%rax, %r14
	testq	%r12, %r12
	je	.LBB2_50
# %bb.49:
	subq	%r12, %rbx
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movq	%rbx, %rsi
	callq	_ZdlPvm
.LBB2_50:
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table2:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    # >> Call Site 1 <<
	.uleb128 .Ltmp0-.Lfunc_begin0           #   Call between .Lfunc_begin0 and .Ltmp0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 2 <<
	.uleb128 .Ltmp1-.Ltmp0                  #   Call between .Ltmp0 and .Ltmp1
	.uleb128 .Ltmp4-.Lfunc_begin0           #     jumps to .Ltmp4
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp5-.Lfunc_begin0           # >> Call Site 3 <<
	.uleb128 .Ltmp6-.Ltmp5                  #   Call between .Ltmp5 and .Ltmp6
	.uleb128 .Ltmp9-.Lfunc_begin0           #     jumps to .Ltmp9
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp10-.Lfunc_begin0          # >> Call Site 4 <<
	.uleb128 .Ltmp11-.Ltmp10                #   Call between .Ltmp10 and .Ltmp11
	.uleb128 .Ltmp14-.Lfunc_begin0          #     jumps to .Ltmp14
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp11-.Lfunc_begin0          # >> Call Site 5 <<
	.uleb128 .Ltmp2-.Ltmp11                 #   Call between .Ltmp11 and .Ltmp2
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp2-.Lfunc_begin0           # >> Call Site 6 <<
	.uleb128 .Ltmp3-.Ltmp2                  #   Call between .Ltmp2 and .Ltmp3
	.uleb128 .Ltmp4-.Lfunc_begin0           #     jumps to .Ltmp4
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp7-.Lfunc_begin0           # >> Call Site 7 <<
	.uleb128 .Ltmp8-.Ltmp7                  #   Call between .Ltmp7 and .Ltmp8
	.uleb128 .Ltmp9-.Lfunc_begin0           #     jumps to .Ltmp9
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp12-.Lfunc_begin0          # >> Call Site 8 <<
	.uleb128 .Ltmp13-.Ltmp12                #   Call between .Ltmp12 and .Ltmp13
	.uleb128 .Ltmp14-.Lfunc_begin0          #     jumps to .Ltmp14
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp13-.Lfunc_begin0          # >> Call Site 9 <<
	.uleb128 .Lfunc_end2-.Ltmp13            #   Call between .Ltmp13 and .Lfunc_end2
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._Z32__device_stub__bandwidth_memreadILi8EEvPKDv4_fPfm,"axG",@progbits,_Z32__device_stub__bandwidth_memreadILi8EEvPKDv4_fPfm,comdat
	.weak	_Z32__device_stub__bandwidth_memreadILi8EEvPKDv4_fPfm # -- Begin function _Z32__device_stub__bandwidth_memreadILi8EEvPKDv4_fPfm
	.p2align	4
	.type	_Z32__device_stub__bandwidth_memreadILi8EEvPKDv4_fPfm,@function
_Z32__device_stub__bandwidth_memreadILi8EEvPKDv4_fPfm: # @_Z32__device_stub__bandwidth_memreadILi8EEvPKDv4_fPfm
	.cfi_startproc
# %bb.0:
	subq	$104, %rsp
	.cfi_def_cfa_offset 112
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdx, 56(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$_Z17bandwidth_memreadILi8EEvPKDv4_fPfm, %edi
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end3:
	.size	_Z32__device_stub__bandwidth_memreadILi8EEvPKDv4_fPfm, .Lfunc_end3-_Z32__device_stub__bandwidth_memreadILi8EEvPKDv4_fPfm
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4                               # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	__hip_gpubin_handle_7037446ac5eaa0ac(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB4_2
# %bb.1:
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle_7037446ac5eaa0ac(%rip)
.LBB4_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z10reduce_sumPKDv4_fPdm, %esi
	movl	$.L__unnamed_1, %edx
	movl	$.L__unnamed_1, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z11fill_kernelPDv4_fm, %esi
	movl	$.L__unnamed_2, %edx
	movl	$.L__unnamed_2, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z17bandwidth_memreadILi8EEvPKDv4_fPfm, %esi
	movl	$.L__unnamed_3, %edx
	movl	$.L__unnamed_3, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	movl	$__hip_module_dtor, %edi
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	atexit                          # TAILCALL
.Lfunc_end4:
	.size	__hip_module_ctor, .Lfunc_end4-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_7037446ac5eaa0ac(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB5_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle_7037446ac5eaa0ac(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB5_2:
	retq
.Lfunc_end5:
	.size	__hip_module_dtor, .Lfunc_end5-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_Z10reduce_sumPKDv4_fPdm,@object # @_Z10reduce_sumPKDv4_fPdm
	.section	.rodata,"a",@progbits
	.globl	_Z10reduce_sumPKDv4_fPdm
	.p2align	3, 0x0
_Z10reduce_sumPKDv4_fPdm:
	.quad	_Z25__device_stub__reduce_sumPKDv4_fPdm
	.size	_Z10reduce_sumPKDv4_fPdm, 8

	.type	_Z11fill_kernelPDv4_fm,@object  # @_Z11fill_kernelPDv4_fm
	.globl	_Z11fill_kernelPDv4_fm
	.p2align	3, 0x0
_Z11fill_kernelPDv4_fm:
	.quad	_Z26__device_stub__fill_kernelPDv4_fm
	.size	_Z11fill_kernelPDv4_fm, 8

	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"HIP error %s at %s:%d\n"
	.size	.L.str, 23

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"bandwidth_memread.hip"
	.size	.L.str.1, 22

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"Device: %s (%s), %d CUs, %.0f MHz\n"
	.size	.L.str.2, 35

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"\nSelf-check: sum=%.0f ref=%.0f rel_err=%.3e -> %s\n"
	.size	.L.str.3, 51

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"PASS"
	.size	.L.str.4, 5

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"FAIL"
	.size	.L.str.5, 5

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"\n%-12s %-12s %-10s\n"
	.size	.L.str.7, 20

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"size(MiB)"
	.size	.L.str.8, 10

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	"iters"
	.size	.L.str.9, 6

	.type	.L.str.10,@object               # @.str.10
.L.str.10:
	.asciz	"GB/s"
	.size	.L.str.10, 5

	.type	.L__const.main.sizes_mib,@object # @__const.main.sizes_mib
	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	4, 0x0
.L__const.main.sizes_mib:
	.quad	64                              # 0x40
	.quad	256                             # 0x100
	.quad	1024                            # 0x400
	.quad	2048                            # 0x800
	.size	.L__const.main.sizes_mib, 32

	.type	.L.str.11,@object               # @.str.11
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.11:
	.asciz	"%-12zu (skip: alloc failed)\n"
	.size	.L.str.11, 29

	.type	_Z17bandwidth_memreadILi8EEvPKDv4_fPfm,@object # @_Z17bandwidth_memreadILi8EEvPKDv4_fPfm
	.section	.rodata._Z17bandwidth_memreadILi8EEvPKDv4_fPfm,"aG",@progbits,_Z17bandwidth_memreadILi8EEvPKDv4_fPfm,comdat
	.weak	_Z17bandwidth_memreadILi8EEvPKDv4_fPfm
	.p2align	3, 0x0
_Z17bandwidth_memreadILi8EEvPKDv4_fPfm:
	.quad	_Z32__device_stub__bandwidth_memreadILi8EEvPKDv4_fPfm
	.size	_Z17bandwidth_memreadILi8EEvPKDv4_fPfm, 8

	.type	.L.str.12,@object               # @.str.12
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.12:
	.asciz	"%-12zu %-12d %-10.0f\n"
	.size	.L.str.12, 22

	.type	.L.str.14,@object               # @.str.14
.L.str.14:
	.asciz	"cannot create std::vector larger than max_size()"
	.size	.L.str.14, 49

	.type	.L__unnamed_1,@object           # @0
.L__unnamed_1:
	.asciz	"_Z10reduce_sumPKDv4_fPdm"
	.size	.L__unnamed_1, 25

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"_Z11fill_kernelPDv4_fm"
	.size	.L__unnamed_2, 23

	.type	.L__unnamed_3,@object           # @2
.L__unnamed_3:
	.asciz	"_Z17bandwidth_memreadILi8EEvPKDv4_fPfm"
	.size	.L__unnamed_3, 39

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_7037446ac5eaa0ac
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_7037446ac5eaa0ac,@object # @__hip_gpubin_handle_7037446ac5eaa0ac
	.local	__hip_gpubin_handle_7037446ac5eaa0ac
	.comm	__hip_gpubin_handle_7037446ac5eaa0ac,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_7037446ac5eaa0ac,@object # @__hip_cuid_7037446ac5eaa0ac
	.bss
	.globl	__hip_cuid_7037446ac5eaa0ac
__hip_cuid_7037446ac5eaa0ac:
	.byte	0                               # 0x0
	.size	__hip_cuid_7037446ac5eaa0ac, 1

	.type	.Lstr,@object                   # @str
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lstr:
	.asciz	"RESULT: FAIL"
	.size	.Lstr, 13

	.type	.Lstr.15,@object                # @str.15
.Lstr.15:
	.asciz	"\nRESULT: PASS"
	.size	.Lstr.15, 14

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z25__device_stub__reduce_sumPKDv4_fPdm
	.addrsig_sym _Z26__device_stub__fill_kernelPDv4_fm
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _Z32__device_stub__bandwidth_memreadILi8EEvPKDv4_fPfm
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _Z10reduce_sumPKDv4_fPdm
	.addrsig_sym _Z11fill_kernelPDv4_fm
	.addrsig_sym _Z17bandwidth_memreadILi8EEvPKDv4_fPfm
	.addrsig_sym __hip_fatbin_7037446ac5eaa0ac
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_7037446ac5eaa0ac

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-

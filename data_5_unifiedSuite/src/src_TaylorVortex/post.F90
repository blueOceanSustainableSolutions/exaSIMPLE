!***********************************************************************
!>
!!     User-defined module for extra post-processing
!!
!!     Maritime Research Institute of Netherlands (MARIN)
!!
!!     ReFRESCO
!!
!!     $Id: post.F90 14303 2022-08-31 12:19:17Z avdploeg@MARIN.LOCAL $
!!
!!     Maintainer: Menno Deij (Guilherme Vaz)
!!     Level:      Infrastructure/User-defined-routines
!!
!!     (c) 2005-2022 MARIN
!!     Proprietary  data. Unauthorized use, distribution,
!!     or duplication is prohibited. All rights reserved.
!!
!<
!***********************************************************************
!
!!      module post
!        this module can be used to set up special post-processing routines. Use
!          -post_outit to modify values in each outer iteration
!          -post_timestep to modify values in each timestep
!          -post_final to set values after the solution 
!           algorithm is finished or stopped
!
!***********************************************************************
MODULE post

USE controls_parameters
USE main_controls
USE equations
USE equations_controls
USE fielddata
USE fielddata_tools
USE geometry
USE parameters
USE topology
USE tracing
USE files
USE parallel
#ifdef ENABLE_OVERSET
USE, INTRINSIC :: iso_fortran_env , only : int32
USE overset
#endif

IMPLICIT NONE

PRIVATE

INTEGER,           PRIVATE :: nAllCells,nIntCells,NBndCells,nIntFaces,nBndFaces,nfamilies
INTEGER                    :: unit_errors,unit_errorsLoc
REAL(dp), POINTER, PRIVATE :: cent_c_3(:,:), vol_c(:)
REAL(dp), POINTER, PRIVATE :: v_c_3(:,:), p_c(:), C_c(:)
REAL(dp), POINTER, PRIVATE :: grad_v_c_33(:,:,:), grad_p_c_3(:,:)
REAL(dp), POINTER, PRIVATE :: exact_v_c_3(:,:), exact_p_c(:), exact_C_c(:)
REAL(dp), POINTER, PRIVATE :: error_v_c_3(:,:), error_p_c(:)
REAL(dp), POINTER, PRIVATE :: exact_grad_v_c_33(:,:,:), exact_grad_p_c_3(:,:)
REAL(dp), POINTER, PRIVATE :: error_grad_v_c_33(:,:,:), error_grad_p_c_3(:,:)
REAL(dp), POINTER, PRIVATE :: hessian_p_c_33(:,:,:),exact_hessian_p_c_33(:,:,:)
	
PUBLIC post_initial,&
       post_init3,&
       post_init4,&
       post_init5,&
       post_restart,&
       post_timestep,&
       post_adaptit,&       
       post_outit,&
       post_final,&
       post_exit,&
       post_read_controls

CONTAINS

!======================================================================
SUBROUTINE post_read_controls
!======================================================================
		
  CALL tracing_trace_begin(trace_post_read_controls)
		
  CALL tracing_trace_end(trace_post_read_controls)

END SUBROUTINE post_read_controls				

!!  use this routine to set up linked lists, to get sizes or to allocate fields
!======================================================================
SUBROUTINE post_init3
!======================================================================

  CALL tracing_trace_begin(trace_post_init3)

  CALL fielddata_register_field("ErrorVelocity",   &
                              ndim = 3,             &
                              toSave=.TRUE.,        &
                              hasField =.TRUE.,     &
                              hasBoundary =.TRUE.,  &
                              hasGradients=.TRUE.,  &
                              toSaveGradients=.TRUE.)

  CALL fielddata_register_field("ErrorPressure",   &
                              ndim = 1,             &
                              toSave=.TRUE.,        &
                              hasField =.TRUE.,     &
                              hasBoundary =.TRUE.,  &
                              hasGradients=.TRUE.,  &
                              toSaveGradients=.TRUE.)

  CALL tracing_trace_end(trace_post_init3)

END SUBROUTINE post_init3
		
!>  use *_init4 module for fielddata_init4,
!<  equation_system_register_equation_system ... statements 
!======================================================================
SUBROUTINE post_init4
!=====================================================================

  CALL tracing_trace_begin(trace_post_init4)

      
  CALL tracing_trace_end(trace_post_init4)
   
END SUBROUTINE post_init4

!! use *_init5 module for all *_get_pointer statements
!======================================================================
SUBROUTINE post_init5
!=====================================================================
	
  CALL tracing_trace_begin(trace_post_init5)

  CALL topology_get_size( nAllCells = nAllCells, &
                          nIntCells = nIntCells, &
                          NBndCells = NBndCells,&
                          nIntFaces = nIntFaces,&
                          nBndFaces = nBndFaces,&
                          nfamilies = nfamilies)

  CALL geometry_get_pointer(cent_c_3=cent_c_3)
  CALL geometry_get_pointer(vol_c=vol_c)
 
	CALL fielddata_get_pointer("Velocity",        &
                           Field_3=v_c_3,     &
                           Grad_33=grad_v_c_33)
	CALL fielddata_get_pointer("ExactVelocity",         &
                           Field_3=exact_v_c_3,     &
                           Grad_33=exact_grad_v_c_33)

  IF (controls%equations%equation(equation_loc_pressure)%EQPressure%saveHessian) THEN
	CALL fielddata_get_pointer("Pressure",        &
                            Field=p_c,        &
                            Grad_3=grad_p_c_3,        &
                            Hessian_33=hessian_p_c_33  )
	CALL fielddata_get_pointer("ExactPressure",         &
                            Field=exact_p_c,        &
                            Grad_3=exact_grad_p_c_3,         &
                            Hessian_33=exact_hessian_p_c_33 )
  ELSE
	CALL fielddata_get_pointer("Pressure",        &
                            Field=p_c,        &
                            Grad_3=grad_p_c_3 )
	CALL fielddata_get_pointer("ExactPressure",         &
                            Field=exact_p_c,        &
                            Grad_3=exact_grad_p_c_3 )
  END IF

  IF (solve_freesurface) THEN
    CALL fielddata_get_pointer("AirVolumeFraction",Field=C_c)
    CALL fielddata_get_pointer("ExactAirVolumeFraction", Field=exact_C_c)
  END IF

  CALL fielddata_get_pointer("ErrorVelocity",         &
                           Field_3=error_v_c_3,     &
                           Grad_33=error_grad_v_c_33)
	CALL fielddata_get_pointer("ErrorPressure",         &
                            Field=error_p_c,        &
                            Grad_3=error_grad_p_c_3 )

  IF (OnMaster) THEN
    CALL files_register_file(fileBaseName = "errors",fileExtension=".dat",fileunit=unit_errors)
    CALL files_open_file(fileBaseName = "errors",baseunit=unit_errors)

    CALL files_register_file(fileBaseName = "errorsLoc",fileExtension=".dat",fileunit=unit_errorsLoc)
    CALL files_open_file(fileBaseName = "errorsLoc",baseunit=unit_errorsLoc)
  ENDIF

  CALL tracing_trace_end(trace_post_init5)

END SUBROUTINE post_init5

!>
!! use this to set initial values. To resolve dependencies between 
!! several *_initial modules this is called several times. Thus 
!! do not put ALLOCATE statements here. 
!<
!======================================================================
SUBROUTINE post_initial
!=====================================================================
		
  CALL tracing_trace_begin(trace_post_initial)
		
  CALL tracing_trace_end(trace_post_initial)

END SUBROUTINE post_initial
    
!!  use *_restart routines for restart
!======================================================================
SUBROUTINE post_restart
!======================================================================
    
  CALL tracing_trace_begin(trace_post_restart)
    
  CALL tracing_trace_end(trace_post_restart)

END SUBROUTINE post_restart

!! this routine is executed before each adapt iteration
!======================================================================
SUBROUTINE post_adaptit(adapt_iter)
!======================================================================
INTEGER, INTENT(IN) :: adapt_iter

  IF (.false.) PRINT *, STORAGE_SIZE(adapt_iter) ! avoid unused warning
		
  CALL tracing_trace_begin(trace_post_adaptit)
		
  CALL tracing_trace_end(trace_post_adaptit)

END SUBROUTINE post_adaptit
		
!! this routine is executed after each timestep
!======================================================================
SUBROUTINE post_timestep(timestep,simultime)
!======================================================================
INTEGER, INTENT(IN)  :: timestep
REAL(dp), INTENT(IN) :: simultime
		
  IF (.false.) PRINT *, STORAGE_SIZE(timestep) ! avoid unused warning
  IF (.false.) PRINT *, STORAGE_SIZE(simultime) ! avoid unused warning

  CALL tracing_trace_begin(trace_post_timestep)
		
  CALL tracing_trace_end(trace_post_timestep)

END SUBROUTINE post_timestep
		
!! this routine is executed after each outer iteration
!======================================================================
SUBROUTINE post_outit(out_iter)
!======================================================================
INTEGER,INTENT(IN) :: out_iter

INTEGER   :: icell

  IF (.false.) PRINT *, STORAGE_SIZE(out_iter) ! avoid unused warning

  CALL tracing_trace_begin(trace_post_outit)

  DO icell = 1, nIntCells
    ! Velocity:
    error_v_c_3(icell,:)        = v_c_3(icell,:) - exact_v_c_3(icell,:)
    ! Velocity gradient:
    error_grad_v_c_33(icell,:,:)= grad_v_c_33(icell,:,:) - exact_grad_v_c_33(icell,:,:)
   
    ! Pressure:
    error_p_c(icell)          = p_c(icell) - exact_p_c(icell)
    ! Pressure gradient:
    error_grad_p_c_3(icell,:) = grad_p_c_3(icell,:) - exact_grad_p_c_3(icell,:)

  END DO

  CALL tracing_trace_end(trace_post_outit)

END SUBROUTINE post_outit

!>
!! this routine is executed only once after the computation has stopped
!! due to max no of iteration, truncation error or stopfile. It is not
!! executed when killfile is used
!<
!======================================================================
SUBROUTINE post_final
!======================================================================
INTEGER                   :: icell,k,nInCells,nSmoothGradientsForPressureHessian,nSmoothPressureHessian
INTEGER                   :: proc,dimens
REAL(DP)                  :: pos(3),dt,area,ff
LOGICAL                   :: hessian
REAL(DP), DIMENSION(14)   :: error_p, error_p_l1, error_p_l2, error_p_linf ! p, dp/dx, dp/dy, dp/dz, 9 components of the hessian and airvolume fraction
REAL(DP), DIMENSION(3,4)  :: error_vel, error_vel_l1, error_vel_l2, error_vel_linf ! velocity components and their gradients

#ifdef ENABLE_OVERSET
integer(int32), pointer, dimension(:) :: IBLANK_c => NULL()  ! IBLANK field data (for overset)
#endif
INTEGER, DIMENSION(14)     :: iproc_p_linf
REAL(DP), DIMENSION(14,3)  :: pos_p_error_linf
INTEGER, DIMENSION(3,4)    :: iproc_vel_linf
REAL(DP), DIMENSION(3,4,3) :: pos_vel_error_linf

  CALL tracing_trace_begin(trace_post_final)

#ifdef ENABLE_OVERSET
    ! Only count InCells in the error computation
  IF (use_overset) THEN
    IF (controls%equations%equation(equation_loc_pressure)%EQPressure%saveHessian) THEN
     nSmoothGradientsForPressureHessian=0
     nSmoothPressureHessian=0
     CALL fielddata_tools_calc_hessian("Pressure",nSmoothGradientsForPressureHessian,nSmoothPressureHessian)
    ENDIF
  END IF
#endif

  ! initializing error norms:
  error_p             = 0.0d0
  error_p_l1          = 0.0d0
  error_p_l2          = 0.0d0
  error_p_linf        = 0.0d0

  error_vel           = 0.0d0
  error_vel_l1        = 0.0d0
  error_vel_l2        = 0.0d0
  error_vel_linf      = 0.0d0

  pos_p_error_linf    = 0.0d0
  pos_vel_error_linf  = 0.0d0
  iproc_p_linf        = 0
  iproc_vel_linf      = 0
  area                = 0.0d0
  nInCells            = 0

#ifdef ENABLE_OVERSET
  IF (use_overset) CALL overset_get_pointer(ptrIBLANK_c=IBLANK_c)
#endif

  hessian = controls%equations%equation(equation_loc_pressure)%EQPressure%saveHessian

  DO icell = 1, nIntCells

#ifdef ENABLE_OVERSET
    ! Only count InCells in the error computation
    IF (use_overset) THEN
      IF (IBLANK_c(icell) /= IN_CELL) CYCLE
    END IF
#endif

    pos = cent_c_3(icell,:)

    ! Error p and derivatives (x,y,z)
    error_p(1)   = p_c(icell) - exact_p_c(icell) 
    error_p(2:4) = grad_p_c_3(icell,:) - exact_grad_p_c_3(icell,:)

    IF (hessian) THEN
      ! error for xx-derivative of p
      error_p(5) = hessian_p_c_33(icell,1,1) - exact_hessian_p_c_33(icell,1,1)
      ! error for xy-, yx- yy-derivative of p
      error_p(6)   = hessian_p_c_33(icell,1,2)   - exact_hessian_p_c_33(icell,1,2)
      error_p(7:8) = hessian_p_c_33(icell,2,1:2) - exact_hessian_p_c_33(icell,2,1:2)
      ! error for xz-, yz-, zx, zy and zz-derivative of p
      error_p(9:10)  = hessian_p_c_33(icell,3,1:2) - exact_hessian_p_c_33(icell,3,1:2)
      error_p(11:13) = hessian_p_c_33(icell,1:3,3) - exact_hessian_p_c_33(icell,1:3,3)
    END IF

    ! Error Airvolume fraction
    error_p(14) = LARGE
    IF (solve_freesurface) error_p(14)   = C_c(icell) - exact_C_c(icell)

    !L1 norm with volume
    error_p_l1 = error_p_l1 + vol_c(icell)*abs(error_p)
    !L2 norm with volume
    error_p_l2 = error_p_l2 + vol_c(icell)*error_p**2  
    ! Linf (max) and their locations
    DO k=1,size(error_p)
      IF ( ABS(error_p(k)) > error_p_linf(k)) THEN
        error_p_linf(k)       = ABS(error_p(k))
        pos_p_error_linf(k,:) = pos
      ENDIF   
    ENDDO
    
    ! Error velocity components and their derivatives (x,y,z)
    error_vel(:,1)   =       v_c_3(icell,:)   - exact_v_c_3(icell,:)  
    error_vel(:,2:4) = grad_v_c_33(icell,:,:) - exact_grad_v_c_33(icell,:,:)
    !L1 norm with volume
    error_vel_l1 = error_vel_l1 + vol_c(icell)*abs(error_vel)
    !L2 norm with volume
    error_vel_l2 = error_vel_l2 + vol_c(icell)*error_vel**2  
    ! Linf (max) and their locations
    DO k=1,4 !loop over p,u,v and w
      DO dimens = 1,3
        IF ( ABS(error_vel(dimens,k)) > error_vel_linf(dimens,k)) THEN
          error_vel_linf(dimens,k)       = ABS(error_vel(dimens,k))
          pos_vel_error_linf(dimens,k,:) = pos
        ENDIF 
      ENDDO  
    ENDDO !loop over p,u,v and w

    area = area + vol_c(icell)    
    nInCells = nInCells + 1
    
  END DO

  ! Final parallel operations
  CALL parallel_sum(nInCells)
  CALL parallel_sum(area)
  DO k=1,size(error_p_l1)
    ! Pressure, pressure gradient and hessian
    CALL parallel_sum(error_p_l1(k))      ! L1 sum
    CALL parallel_sum(error_p_l2(k))      ! L2 sum
    ff = error_p_linf(k)
    CALL parallel_maxloc(ff,proc,error_p_linf(k))                ! Linf max and its location in terms of proc
    iproc_p_linf(k)=proc ; CALL parallel_bcast(pos_p_error_linf(k,:),proc)    ! broadcast linf position
  ENDDO
  DO k=1,4
    CALL parallel_sum(error_vel_l1(1,k))      ! L1 sum of error_u
    CALL parallel_sum(error_vel_l1(2,k))      ! L1 sum of error_v
    CALL parallel_sum(error_vel_l1(3,k))      ! L1 sum of error_w
    CALL parallel_sum(error_vel_l2(1,k))      ! L2 sum of error_u
    CALL parallel_sum(error_vel_l2(2,k))      ! L2 sum of error_v
    CALL parallel_sum(error_vel_l2(3,k))      ! L2 sum of error_w
    DO dimens = 1,3
      ff = error_vel_linf(dimens,k)
      CALL parallel_maxloc(ff,proc,error_vel_linf(dimens,k))                ! Linf max and its location in terms of proc
      iproc_vel_linf(dimens,k)=proc ; CALL parallel_bcast(pos_vel_error_linf(dimens,k,:),proc)  ! broadcast linf position
    ENDDO
  END DO
  error_p_l1   = error_p_l1/area
  error_vel_l1 = error_vel_l1/area
  error_p_l2   = SQRT(error_p_l2)/sqrt(area)
  error_vel_l2 = SQRT(error_vel_l2)/sqrt(area)
  dt = controls%timeLoop%timeDelta

  ! Writing
  IF (OnMaster) THEN ! Compatible with order program verificationSuite/data/order.F90:
                     !   first quantities  u,v,w and p 
                     !   then du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz, dw/dx, dw/dy, dw/dz
                     !   then the pressure gradient, 
                     !   then the airvolumefraction
                     !   and finally the hessian
    IF (hessian) THEN
      WRITE(unit_errors, '(I8,1P,27E25.12,A)') nInCells, dt, &
             error_vel_l1(1,1),  error_vel_l1(2,1),  error_vel_l1(3,1),  error_p_l1(1),  &
             error_vel_l1(1,2:4),error_vel_l1(2,2:4),error_vel_l1(3,2:4),error_p_l1(2:4),  &  
             error_p_l1(14),error_p_l1(5:13), " # error_l1"
      WRITE(unit_errors, '(I8,1P,27E25.12,A)') nInCells, dt, &
             error_vel_l2(1,1),  error_vel_l2(2,1),  error_vel_l2(3,1),  error_p_l2(1),  &
             error_vel_l2(1,2:4),error_vel_l2(2,2:4),error_vel_l2(3,2:4),error_p_l2(2:4),  &  
             error_p_l2(14),error_p_l2(5:13), " # error_l2"
      WRITE(unit_errors, '(I8,1P,27E25.12,A)') nInCells, dt, &
             error_vel_linf(1,1),  error_vel_linf(2,1),  error_vel_linf(3,1),  error_p_linf(1),  &
             error_vel_linf(1,2:4),error_vel_linf(2,2:4),error_vel_linf(3,2:4),error_p_linf(2:4),  &  
             error_p_linf(14),error_p_linf(5:13), " # error_linf"
    ELSE
      WRITE(unit_errors, '(I8,1P,18E25.12,A)') nInCells, dt, &
             error_vel_l1(1,1),  error_vel_l1(2,1),  error_vel_l1(3,1),  error_p_l1(1),  &
             error_vel_l1(1,2:4),error_vel_l1(2,2:4),error_vel_l1(3,2:4),error_p_l1(2:4), &
             error_p_l1(14), " # error_l1"
      WRITE(unit_errors, '(I8,1P,18E25.12,A)') nInCells, dt, &
             error_vel_l2(1,1),  error_vel_l2(2,1),  error_vel_l2(3,1),  error_p_l2(1),  &
             error_vel_l2(1,2:4),error_vel_l2(2,2:4),error_vel_l2(3,2:4),error_p_l2(2:4), &
             error_p_l2(14), " # error_l2"
      WRITE(unit_errors, '(I8,1P,18E25.12,A)') nInCells, dt, &
             error_vel_linf(1,1),  error_vel_linf(2,1),  error_vel_linf(3,1),  error_p_linf(1),  &
             error_vel_linf(1,2:4),error_vel_linf(2,2:4),error_vel_linf(3,2:4),error_p_linf(2:4), &
             error_p_linf(14), " # error_linf"
    END IF
                                                         
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='p'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_p_error_linf(1,:), error_p_linf(1)
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='dp/dx'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_p_error_linf(2,:), error_p_linf(2)
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='dp/dy'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_p_error_linf(3,:), error_p_linf(3)
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='dp/dz'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_p_error_linf(4,:), error_p_linf(4)
    IF (hessian) THEN
      WRITE(unit_errorsLoc, '(A)')  "ZONE T='dp/dxx'"
      WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_p_error_linf(5,:), error_p_linf(5)
      WRITE(unit_errorsLoc, '(A)')  "ZONE T='dp/dyx'"
      WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_p_error_linf(6,:), error_p_linf(6)
      WRITE(unit_errorsLoc, '(A)')  "ZONE T='dp/dxy'"
      WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_p_error_linf(7,:), error_p_linf(7)
      WRITE(unit_errorsLoc, '(A)')  "ZONE T='dp/dyy'"
      WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_p_error_linf(8,:), error_p_linf(8)
      WRITE(unit_errorsLoc, '(A)')  "ZONE T='dp/dxz'"
      WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_p_error_linf(9,:), error_p_linf(9)
      WRITE(unit_errorsLoc, '(A)')  "ZONE T='dp/dyz'"
      WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_p_error_linf(10,:), error_p_linf(10)
      WRITE(unit_errorsLoc, '(A)')  "ZONE T='dp/dzx'"
      WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_p_error_linf(11,:), error_p_linf(11)
      WRITE(unit_errorsLoc, '(A)')  "ZONE T='dp/dzy'"
      WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_p_error_linf(12,:), error_p_linf(12)
      WRITE(unit_errorsLoc, '(A)')  "ZONE T='dp/dzz'"
      WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_p_error_linf(13,:), error_p_linf(13)
    END IF
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='Airvolumefraction'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_p_error_linf(14,:), error_p_linf(14)
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='u'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_vel_error_linf(1,1,:), error_vel_linf(1,1)     
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='du/dx'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_vel_error_linf(1,2,:), error_vel_linf(1,2)
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='du/dy'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_vel_error_linf(1,3,:), error_vel_linf(1,3)
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='du/dz'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_vel_error_linf(1,4,:), error_vel_linf(1,4)
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='v'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_vel_error_linf(2,1,:), error_vel_linf(2,1)     
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='dv/dx'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_vel_error_linf(2,2,:), error_vel_linf(2,2)
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='dv/dy'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_vel_error_linf(2,3,:), error_vel_linf(2,3)
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='dv/dz'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_vel_error_linf(2,4,:), error_vel_linf(2,4)
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='w'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_vel_error_linf(3,1,:), error_vel_linf(3,1)     
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='dw/dx'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_vel_error_linf(3,2,:), error_vel_linf(3,2)
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='dw/dy'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_vel_error_linf(3,3,:), error_vel_linf(3,3)
    WRITE(unit_errorsLoc, '(A)')  "ZONE T='dw/dz'"
    WRITE(unit_errorsLoc, '(1P,4E25.12)') pos_vel_error_linf(3,4,:), error_vel_linf(3,4)
  ENDIF

  CALL tracing_trace_end(trace_post_final)

END SUBROUTINE post_final

!! use *_exit module for DEALLOCATE statements etc. 
!======================================================================
   SUBROUTINE post_exit
!=====================================================================

   CALL tracing_trace_begin(trace_post_exit)

   CALL tracing_trace_end(trace_post_exit)

END SUBROUTINE post_exit

END MODULE post

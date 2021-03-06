module python_caller
  ! only works with CAM radiation scheme
  use callpy_mod
  implicit none

  public :: initialize_from_target, initialize_python_caller, state_to_python, push_state


  character(len=256) :: module_name, function_name
  logical :: do_debias_lhf = .false.

  real :: last_time_called
  real, allocatable, dimension(:,:,:) :: sl_last, qt_last, FQTNN, FSLINN,&
    funn, fvnn
  real, allocatable :: net_prec_xy(:, :), net_heat_xy(:, :)

  integer ntop
  ! Do not apply neural network within this boundary region
  integer, parameter :: meridional_bndy_size = 3
contains

  subroutine bias_correct_lhf(x, corrected)
    ! Use a polynomial fit to correct the latent heat fluxes
    ! I estimated these coefficients using python
    real, dimension(3), parameter :: coefs  = [-5.24149046e-04,  1.15227747e+00,  4.92990095e+00]
    real, dimension(:, :), intent(in) :: x
    real, dimension(:, :), intent(out) :: corrected

    integer k, n
    corrected = 0.0
    n = size(coefs)
    do k=1,n
       corrected = corrected  + x**(n-k) * coefs(k)
    end do
  end subroutine bias_correct_lhf

  subroutine initialize_python_caller()
    use vars, only: nx, ny, nz, nzm, rho, pres, presi, caseid, case, adz, dz, t, time
    use microphysics, only: micro_field

    ! locals
    real(4) :: tmp(1:nzm), tmpw(1:nz)

    allocate(sl_last(nx, ny, nzm))
    allocate(qt_last(nx, ny, nzm))
    allocate(FQTNN(nx, ny, nzm))
    allocate(FSLINN(nx, ny, nzm))
    allocate(funn(nx, ny, nzm))
    allocate(fvnn(nx, ny, nzm))
    allocate(net_prec_xy(nx, ny))
    allocate(net_heat_xy(nx, ny))

    sl_last = t(1:nx, 1:ny, 1:nzm)
    qt_last = micro_field(1:nx, 1:ny, 1:nzm, 1)
    last_time_called = time
    print *, 'python_caller.f90::initialize_python_caller: storing sl and qt to sl_last and qt_last at time', time

    tmp = rho * adz * dz
    call set_state_1d("layer_mass", tmp)

    tmp = pres
    call set_state_1d("p", tmp)

    tmpw = presi
    call set_state_1d("pi", tmpw)

    call set_state_char("caseid", caseid)
    call set_state_char("case", case)
  end subroutine initialize_python_caller

  subroutine initialize_from_target
    use vars, only: t, u, v,w, tabs,&
         shf_xy, lhf_xy, sstxy, prec_xy,&
         latitude, longitude,&
         nx, ny, nzm, rho, adz, dz, pres, presi, time, nstep
    ! use rad, only: solinxy
    ! use grid, only: day, caseid, case
    use microphysics, only: micro_field
    real(4) :: tmp(1:nx, 1:ny, 1:nzm)
    ! locals

    print *, 'Initializing state from target'

    ! call check(compute_next_state())

    ! read in state from python
    call get_state("sl", tmp, nx * ny * nzm)
    t(1:nx, 1:ny, 1:nzm) = tmp

    call get_state("qt", tmp, nx * ny * nzm)
    micro_field(1:nx, 1:ny, 1:nzm, 1) = tmp

    call get_state("U", tmp, nx*ny*nzm)
    u(1:nx, 1:ny, 1:nzm) = tmp

    call get_state("V", tmp, nx*ny*nzm)
    v(1:nx, 1:ny, 1:nzm) = tmp

    call get_state("W", tmp, nx*ny*nzm)
    w(1:nx, 1:ny, 1:nzm) = tmp

    call get_state("Prec", tmp, nx*ny)
    prec_xy(1:nx, 1:ny) = tmp(:,:,1)

    call get_state("LHF", tmp, nx * ny)
    lhf_xy(1:nx, 1:ny) = tmp(:,:,1)

    last_time_called = time
  end subroutine initialize_from_target

  subroutine push_state()
    use vars, only: t, u, v,w, tabs,&
         shf_xy, lhf_xy, sstxy, t00, prec_xy,&
         latitude, longitude,&
         nx, ny, nzm, rho, adz, dz, pres, presi, time, nstep, day, caseid, case, nz, dtn,&
         fluxbq, fluxbt, rhow
    use microphysics, only: micro_field
    use params, only: lcond, cp
    use rad, only: solinxy
    ! compute derivatives
    real, dimension(nx,ny,nzm) :: fqt, fsl
    real(4) :: tmp_vert(1:nzm), tmp_vertw(1:nz), tmp_scalar
    real(4), dimension(1:nx, 1:ny, 1:nzm) :: tmp
    real, dimension(1:nx, 1:ny) :: lhf_no_bias
    real :: dt

    ntop = nzm
    dt = time - last_time_called
    print *, 'python_caller.f90::push_state computing FSL and FQT with dt=', dt
    fsl = (t(1:nx, 1:ny, 1:nzm) - sl_last)/dt
    fqt = (micro_field(1:nx, 1:ny, 1:nzm, 1) - qt_last)/dt

    ! ! Send the arrays to a global array in the python module
    ! ! This is a much easier architecture than trying to write functions
    ! ! which take all of these arguments
    tmp = t(1:nx,1:ny,1:nzm)
    call set_state("SLI", tmp)

    tmp = micro_field(1:nx,1:ny,1:nzm, 1) * 1.e3
    call set_state("QT", tmp)

    tmp =  tabs(1:nx,1:ny,1:nzm)
    call set_state("TABS", tmp)
    tmp =  w(1:nx,1:ny,1:nzm)
    call set_state("W", tmp)

    tmp =  fqt
    call set_state("FQT", tmp)

    tmp =  fsl
    call set_state("FSLI", tmp)

    tmp =  u(1:nx,1:ny,1:nzm)
    call set_state("U", tmp)

    tmp =  v(1:nx,1:ny,1:nzm)
    call set_state("V", tmp)

    tmp(1:nx,1:ny,1) = latitude
    call set_state2d("lat", tmp(1:nx,1:ny,1))

    tmp(1:nx,1:ny,1) = longitude
    call set_state2d("lon", tmp(1:nx,1:ny,1))

    ! for some reason set_state2d has some extremee side ffects
    ! that can cause the model to crash
    tmp(:,:,1) = sstxy(1:nx, 1:ny) + t00
    call set_state2d("SST", tmp(:,:,1))

    tmp(:,:,1) = solinxy(1:nx, 1:ny)
    call set_state2d("SOLIN", tmp(:,:,1))

    tmp(:,:,1) = fluxbt(1:nx, 1:ny) * cp * rhow(1)
    call set_state2d("SHF", tmp(:,:,1))

    if (do_debias_lhf) then
      call bias_correct_lhf(fluxbq(1:nx, 1:ny) * lcond * rhow(1), lhf_no_bias)
      tmp(:,:,1) = lhf_no_bias
    else
        tmp(:,:,1) = fluxbq(1:nx, 1:ny) * lcond * rhow(1)
    end if
    call set_state2d("LHF", tmp(:,:,1))

    tmp_vert = rho * adz * dz
    call set_state_1d("layer_mass", tmp_vert)

    tmp_vert = pres
    call set_state_1d("p", tmp_vert)

    tmp_vertw = presi
    call set_state_1d("pi", tmp_vertw)


    tmp_scalar = dtn
    call set_state_scalar("p0", tmp_scalar)
    call set_state_scalar("dt", tmp_scalar)

    tmp_scalar = time
    call set_state_scalar("time", tmp_scalar)

    tmp_scalar = day
    call set_state_scalar("day", tmp_scalar)

    tmp_scalar = real(nstep)
    call set_state_scalar("nstep", tmp_scalar)
    call set_state_char("caseid", caseid)
    call set_state_char("case", case)

  end subroutine push_state

  subroutine get_state_from_python()
    use vars, only: t, u, v,w, tabs,&
         shf_xy, lhf_xy, sstxy, t00, prec_xy,&
         latitude, longitude,&
         nx, ny, nzm, rho, adz, dz, pres, presi, time, nstep
    use microphysics, only: micro_field, qn, qp, micro_diagnose
    ! locals
    real(4) :: tmp(1:nx, 1:ny, 1:nzm)

    integer js, jn

    ! get furthest north and south grid points to use NN output from
    js = 1 + meridional_bndy_size
    jn = ny - meridional_bndy_size

    ! read in state from python
    print *, 'python_caller.f90::state_to_python retreiving state from python module'
    call get_state("SLI", tmp, nx * ny * nzm)
    ! tmp(:,:,ntop:nzm) = t(1:nx,1:ny, ntop:nzm)
    t(1:nx, 1:ny, 1:nzm) = tmp

    call get_state("QT", tmp, nx * ny * nzm)
    tmp = tmp / 1.e3
    ! tmp(:,:,ntop:nzm) = micro_field(1:nx,1:ny, ntop:nzm, 1)
    micro_field(1:nx, 1:ny, 1:nzm, 1) = tmp


    call get_state("FQTNN", tmp, nx * ny * nzm)
    ! tmp(:,:,ntop:nzm) = t(1:nx,1:ny, ntop:nzm)
    print *, 'SUM OF FQTNN ', sum(tmp**2)
    fqtnn(1:nx, js:jn, 1:nzm) = tmp(:,js:jn,:)

    call get_state("FSLINN", tmp, nx * ny * nzm)
    ! tmp(:,:,ntop:nzm) = t(1:nx,1:ny, ntop:nzm)
    fslinn(1:nx, js:jn, 1:nzm) = tmp(:,js:jn,:)

    call get_state("FUNN", tmp, nx * ny * nzm)
    ! tmp(:,:,ntop:nzm) = t(1:nx,1:ny, ntop:nzm)
    funn(1:nx, js:jn, 1:nzm) = tmp(:,js:jn,:)

    call get_state("FVNN", tmp, nx * ny * nzm)
    ! tmp(:,:,ntop:nzm) = t(1:nx,1:ny, ntop:nzm)
    fvnn(1:nx, js:jn, 1:nzm) = tmp(:,js:jn,:)

    call get_state("QN", tmp, nx*ny*nzm)
    tmp = tmp / 1.e3
    qn(1:nx, js:jn, 1:nzm) = tmp(:,js:jn,:)

    call get_state("QP", tmp, nx*ny*nzm)
    tmp = tmp / 1.e3
    qp(1:nx, js:jn, 1:nzm) = tmp(:,js:jn,:)

    call get_state("Prec", tmp, nx*ny)
    prec_xy(1:nx, 1:ny) = tmp(:,:,1)

    call get_state("LHF", tmp, nx * ny)
    lhf_xy(1:nx, 1:ny) = tmp(:,:,1)

    call get_state("SHF", tmp, nx * ny)
    shf_xy(1:nx, 1:ny) = tmp(:,:,1)

    ! call get_state("U", tmp, nx*ny*nzm)
    ! u(1:nx, js:jn, 1:nzm) = tmp(:,js:jn,:)

    ! call get_state("V", tmp, nx*ny*nzm)
    ! v(1:nx, js:jn, 1:nzm) = tmp(:,js:jn,:)

    ! call get_state("W", tmp, nx*ny*nzm)
    ! w(1:nx, 1:ny, 1:nzm) = tmp

    ! diagnose the phase partitioning of water
    ! this is needed to compute the correct temperature
    ! diagnose is called in main.f90, so we don't need to call that here
    ! ...I hate all these global variables
    call micro_diagnose()

  end subroutine get_state_from_python

  subroutine state_to_python(dtn)
    use vars, only: t, u, v,w, tabs,&
         shf_xy, lhf_xy, sstxy, t00, prec_xy,&
         latitude, longitude,&
         nx, ny, nzm, rho, adz, dz, pres, presi, time, nstep
    use params, only: npython, usepython
    use grid, only: day, caseid, case
    use microphysics, only: micro_field

    ! arguments
    real :: dtn


    integer k

    call push_state()

    ! Compute the necessary variables
    print *, 'python_caller.f90::state_to_python calling python code'
    call call_function(module_name, function_name)
    call get_state_from_python()

    print *, 'python_caller.f90::state_to_python storing state to sl_last and qt_last at time', time
    sl_last = t(1:nx, 1:ny, 1:nzm)
    qt_last = micro_field(1:nx, 1:ny, 1:nzm, 1)
    last_time_called = time
  end subroutine state_to_python

  subroutine apply_nn_forcings(dt)
    use microphysics, only: micro_field
    use vars, only: t, u, v,w, tabs,&
         latitude, longitude, na, &
         nx, ny, nzm, rho, adz, dz, pres, presi, time, nstep, dudt, dvdt
    use params, only: cp
    real, intent(in) :: dt
    real :: dt_days
    real, parameter :: seconds_in_day = 86400.0, gkg_to_kgkg=1000.0
    real :: layer_mass(nzm)

    dt_days = dt / seconds_in_day
    layer_mass = dz * adz * rho

    t(1:nx, 1:ny, 1:nzm) = t(1:nx, 1:ny, 1:nzm) &
         + fslinn(1:nx, 1:ny, 1:nzm) * dt_days
    micro_field(1:nx, 1:ny, 1:nzm, 1) = micro_field(1:nx, 1:ny, 1:nzm, 1) &
         + fqtnn(1:nx, 1:ny, 1:nzm) * dt_days  / gkg_to_kgkg

    dudt(1:nx, 1:ny, 1:nzm,na) = dudt(1:nx, 1:ny, 1:nzm,na) + funn * dt_days
    dvdt(1:nx, 1:ny, 1:nzm,na) = dvdt(1:nx, 1:ny, 1:nzm,na) + fvnn * dt_days

    ! mm / day
    net_prec_xy = - vertical_sum(fqtnn, layer_mass) / gkg_to_kgkg
    ! W/m2
    net_heat_xy = vertical_sum(fslinn, layer_mass) * cp / seconds_in_day
  end subroutine apply_nn_forcings

  function vertical_sum(x, w)
    real, intent(in) :: x(:,:,:), w(:)
    real :: vertical_sum(size(x, 1), size(x, 2))

    integer i, nz
    vertical_sum = 0.0

    nz = size(x, 3)

    do i=1,nz
       vertical_sum = vertical_sum + w(i)*x(:,:,i)
    end do

  end function vertical_sum

end module python_caller

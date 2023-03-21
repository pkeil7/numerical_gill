using DrWatson
@quickactivate "numerical_gill" # <- project name
using NetCDF
using DistributedArrays
using Distributed
include("models.jl")


struct Constructor{T}
    #xs,ys,Xs,Ys,x_dim,y_dim,N,Δt,Δx,Δy,u,v,p,Q

    xs::Vector{T} # vector of x values
    ys::Vector{T} # vector of y values
    Xs::Matrix{T} # array of x values
    Ys::Matrix{T} # array of y values
    x_dim::Int # x_dimension
    y_dim::Int # y dimension
    N::Int # time dimension
    Δt::T
    Δx::T
    Δy::T
    u0::Matrix{T}
    v0::Matrix{T}
    p0::Matrix{T}
    Q::Matrix{T}

end

struct helper{T}
    x::Array{T,3}
    x_dummy::Array{T,3}
    x_old::Array{T,3}
    fs::Array{T,3}
    k1::Array{T,3}
    k2::Array{T,3}
    k3::Array{T,3}
    k4::Array{T,3}
end

function make_helper(x_dim,y_dim)
    return helper{Float64}(
    zeros(3,x_dim,y_dim),
    zeros(3,x_dim,y_dim),
    zeros(3,x_dim,y_dim),
    zeros(3,x_dim,y_dim),
    zeros(3,x_dim,y_dim),
    zeros(3,x_dim,y_dim),
    zeros(3,x_dim,y_dim),
    zeros(3,x_dim,y_dim)
    )
end


function runge_kutta(f::Function,dummies::helper, constants)
    # we need a dummy copy of the state x in the constructor

    h = constants[3]
    f(dummies.k1,dummies.x,constants) # this sets k1
    @. dummies.x_dummy = dummies.x + h/2 * dummies.k1 # this calculates x dummy with k1
    f(dummies.k2, dummies.x_dummy, constants) # this sets k2 with new x
    @. dummies.x_dummy = dummies.x + h/2 * dummies.k2 # this calculates new x dummy with k2 etc.
    f(dummies.k3, dummies.x_dummy, constants)
    @. dummies.x_dummy = dummies.x + h * dummies.k3
    f(dummies.k4, dummies.x_dummy, constants)
    return h/6 * ( dummies.k1 + 2*dummies.k2 + 2*dummies.k3 + dummies.k4)
end


function solver(con::Constructor,f,ϵ_u=0.1, ϵ_v=0.1, ϵ_p=0.1,boundary_condition=cyclical_lon_BC )

    N = con.N
    u = zeros(con.x_dim,con.y_dim,con.N)
    v = zeros(con.x_dim,con.y_dim,con.N)
    p = zeros(con.x_dim,con.y_dim,con.N)

    dummies = make_helper(con.x_dim,con.y_dim)

    constants = (con.x_dim, con.y_dim, con.Δt, con.Δx, con.Δy, con.ys, con.Q, ϵ_u, ϵ_v, ϵ_p)

    x = dummies.x
    fs = dummies.fs
    x_old = dummies.x_old

    for n in 1:N-1
        fs = runge_kutta(f, dummies, constants)
        x[1,:,:] .= view(fs,1,:,:) .+ view(u,:,:,n)
        x[2,:,:] .= view(fs,2,:,:) .+ view(v,:,:,n)
        x[3,:,:] .= view(fs,3,:,:) .+ view(p,:,:,n)

        # fu,fv,fp are zero at the edges, so we need to apply Boundary conditions
        x = boundary_condition(x,con)
        # set new system state for next runge kutta step
        u[:,:,n+1] .= view(x,1,:,:)
        v[:,:,n+1] .= view(x,2,:,:)
        p[:,:,n+1] .= view(x,3,:,:)

    end
    return u,v,p
end


function solver_converge(con::Constructor,f,ϵ_u=0.1, ϵ_v=0.1, ϵ_p=0.1; stop_del=1e-5, save_time=false, save_interval = 20, boundary_condition=cyclical_lon_BC,print_info=true)
    # stop_del should be at least 1e-5 I think. For stop_del = 1e-7 under "normal" paramters, simulation does not converge.
    # so maybe 1e-5 > stop_del > 1e-7

    final_n = 0
    go = true
    n = 1
    convergence = false
    stop_n = con.N

    dummies = make_helper(con.x_dim,con.y_dim)

    constants = (con.x_dim, con.y_dim, con.Δt, con.Δx, con.Δy, con.ys, con.Q, ϵ_u, ϵ_v, ϵ_p)

    x = dummies.x
    fs = dummies.fs
    x_old = dummies.x_old
    if save_time
        x_save = zeros(3, con.x_dim, con.y_dim, Int(stop_n/save_interval) )
    end

    @time while go
        fs = runge_kutta(f, dummies, constants)
        x[1,:,:] .= view(fs,1,:,:) .+ view(x_old,1,:,:)
        x[2,:,:] .= view(fs,2,:,:) .+ view(x_old,2,:,:)
        x[3,:,:] .= view(fs,3,:,:) .+ view(x_old,3,:,:)

        # fu,fv,fp are zero at the edges, so we need to apply Boundary conditions
        x = boundary_condition(x,con)

        #save some timesteps
        if save_time && mod(n+1,save_interval) == 0
            i = Int((n+1)/save_interval)
            x_save[:,:,:,i] = view(x,:,:,:)
        end

        # evaluate stationarity criterion:
        delta_p = view(x,3,:,:) .- view(x_old,3,:,:)
        mean = sum(delta_p)/(con.x_dim*con.y_dim)
        if ( abs(mean) < stop_del && maximum(abs.(delta_p)) < stop_del*10 )
            final_n = n+1
            if print_info
                println("Simulated $final_n timesteps")
            end
            go = false
            convergence = true
        elseif n+1 == stop_n
            final_n = n+1
            if print_info
                println("Reached final simulation timestep, p did not meet stationarity condition! ")
            end
            go = false
        else
            n=n+1
        end

        # set new system state for next runge kutta step
        x_old[1,:,:] .= view(x,1,:,:)
        x_old[2,:,:] .= view(x,2,:,:)
        x_old[3,:,:] .= view(x,3,:,:)

    end

    if save_time
        return x_save[1,:,:,1:floor(Int, final_n/save_interval)],x_save[2,:,:,1:floor(Int, final_n/save_interval)],x_save[3,:,:,1:floor(Int, final_n/save_interval)], convergence
    else
        return x[1,:,:],x[2,:,:],x[3,:,:], convergence
    end

end

function cyclical_lon_BC(x::Array,con::Constructor)
    # This is cyclical in x direction, and zero gradient in y direction

    # x direction
    x[:,1,:] = view(x,:,con.x_dim-1,:)
    x[:,con.x_dim,:] = view(x,:,2,:)

    # y direction
    # y_BC_damp determines the damping in y direction at the edges.
    # y_BC_damp = 1, no damping, variables assumed to be constant accros y- Boundary
    # y_BC_damp = 0, maximum damping, var goes straght to zero at boundary. This often causes some artificial waves close to boundaries!

    y_BC_damp=1
    x[:,:,1] .= y_BC_damp .* view(x,:,:,2)
    x[:,:,con.y_dim] .= y_BC_damp .* view(x,:,:,con.y_dim-1)

    return x
end

function cyclical_lon_BC_2(x::Array,con::Constructor)
    # This is cyclical in x direction, and zero gradient in y direction

    # x direction
    x[:,2,:] = view(x,:,con.x_dim-2,:)
    x[:,1,:] = view(x,:,con.x_dim-3,:)

    x[:,con.x_dim,:] = view(x,:,4,:)
    x[:,con.x_dim-1,:] = view(x,:,3,:)

    # y direction
    # y_BC_damp determines the damping in y direction at the edges.
    # y_BC_damp = 1, no damping, variables assumed to be constant accros y- Boundary
    # y_BC_damp = 0, maximum damping, var goes straght to zero at boundary. This often causes some artificial waves close to boundaries!

    y_BC_damp=1
    x[:,:,1] .= y_BC_damp .* view(x,:,:,3)
    x[:,:,2] .= y_BC_damp .* view(x,:,:,3)

    x[:,:,con.y_dim] .= y_BC_damp .* view(x,:,:,con.y_dim-2)
    x[:,:,con.y_dim-1] .= y_BC_damp .* view(x,:,:,con.y_dim-2)

    return x
end

function zero_gradient_BC(x::Array,con::Constructor)
    # Boundary condition function
    # zero gradient in x and y direction
    # x direction
    x[:,1,:] = view(x,:,2,:)
    x[:,con.x_dim,:] = view(x,:,con.x_dim-1,:)

    # y direction
    x[:,:,1] = view(x,:,:,2)
    x[:,:,con.y_dim] = view(x,:,:,con.y_dim-1)
    return x
end

function construct(Lx,Ly,Δx,Δy,N,Δt,input_Q=1,u0v=0, v0v=0, p0v=0)
    xs=float(0:Δx:Lx)
    ys=float(-Ly:Δy:Ly)

    x_dim = length(xs)
    y_dim = length(ys)


    Xs = repeat(reshape(xs, 1, :), y_dim, 1)
    Ys = repeat(ys, 1, x_dim)

    # --- this does not do anything and can probably be deleted at some point:
    u0 = zeros(x_dim,y_dim) + ones(x_dim,y_dim) * u0v
    v0 = zeros(x_dim,y_dim) + ones(x_dim,y_dim) * v0v
    p0 = zeros(x_dim,y_dim) + ones(x_dim,y_dim) * p0v
    # ---

    # input_Q can be 1,2,3,4 for idealised cases or a custom array, which should be 2 indices smaller in each dimension

    if input_Q == 1
        Q = transpose(Q1.(Xs,Ys))
    elseif input_Q == 2
        Q = transpose(Q2.(Xs,Ys))
    elseif input_Q == 3
        Q = transpose(Q3.(Xs,Ys))
    elseif input_Q == 4
        Q = transpose(Q1.(Xs,Ys)) + transpose(Q2.(Xs,Ys))
    else
        # Given Q has slightly smaller dimension sizes, we add zeros at the edges
        @assert size(input_Q)[1] + 2 == size(u0)[1]
        @assert size(input_Q)[2] + 2 == size(u0)[2]

        Q = zeros(size(u0)[1],size(u0)[2])
        Q[2:(x_dim-1),2:(y_dim-1)] = input_Q

    end

    return Constructor{Float64}(xs,ys,Xs,Ys,x_dim,y_dim,N,Δt,Δx,Δy,u0,v0,p0,Q)
end


function saver(name::String,u,v,p,xs,ys,f; ϵ_u=0.1, ϵ_v=0.1, ϵ_p=0.1, res=0.5,print_info=true,add_attributes=Dict())
    # use dr watson savename
    epsilon_u, epsilon_v, epsilon_p = ϵ_u, ϵ_v, ϵ_p
    params = @strdict f epsilon_u epsilon_v epsilon_p res
    save_name = savename(params,"nc")
    filename = datadir() * "/" * name * "_" * save_name
    # save all in one nc file
    if print_info
        print("Saving simulation as $filename")
    end
    run(`rm -f $filename`)

    standard_attributes = Dict(
    "epsilon_u"   => epsilon_u,
    "epsilon_v"   => epsilon_v,
    "epsilon_p"   => epsilon_p,
    "f" => f
    )

    attributes = merge(standard_attributes, add_attributes)

    nccreate(filename, "u", "x", xs, "y", ys)
    nccreate(filename, "v", "x", xs, "y", ys)
    nccreate(filename, "p", "x", xs, "y", ys, atts=attributes)

    NetCDF.ncwrite(u, filename, "u")
    NetCDF.ncwrite(v, filename, "v")
    NetCDF.ncwrite(p, filename, "p")

    ncclose(filename)
end

function force_balance_gill(con::Constructor,u::Array,v::Array,p::Array,ϵ_u::AbstractFloat, ϵ_v::AbstractFloat, ϵ_p::AbstractFloat)
    #  calculates the individual temers of the gill model for a certain time. (u,v,p must be 2D)
    Ys = transpose(con.Ys)
    Δx = con.Δx
    Δy = con.Δy

    fu_1 = - ϵ_u .* u
    fu_2 = 1/2 .* Ys .* v
    fu_3 = zeros(size(fu_2))
    fp_2 = zeros(size(fu_1))


    for i in 2:con.x_dim-1
        fu_3[i,:] = - (p[i+1,:]-p[i-1,:])/2Δx
        fp_2[i,:] = - (u[i+1,:]-u[i-1,:])/2Δx
    end
    fv_1 = - ϵ_u .* v
    fv_2 = - 1/2 .* Ys .* u
    fv_3 = zeros(size(fv_2))
    fp_3 = zeros(size(fv_2))

    for j in 2:con.y_dim-1
        fv_3[:,j] = - (p[:,j+1]-p[:,j-1])/2Δy
        fp_3[:,j] = - (v[:,j+1]-v[:,j-1])/2Δy
    end

    fp_1 = - ϵ_p .* p

    return [fu_1,fu_2,fu_3],[fv_1,fv_2,fv_3],[fp_1,fp_2,fp_3,con.Q]
end

function geostrophic_winds(con::Constructor,p::Array)
    #  calculates the individual temers of the gill model for a certain time. (u,v,p must be 2D)
    ys = con.ys
    Δx = con.Δx
    Δy = con.Δy

    v_geo = zeros(size(p))
    u_geo = zeros(size(p))


    for i in 2:con.x_dim-1
        v_geo[i,:] .=  2 .* 1 ./ ys .* (p[i+1,:] .- p[i-1,:]) ./ 2Δx
    end
    for i in 2:con.y_dim-1
        u_geo[:,i] .= - 2 * 1 ./ ys[i] * (p[:,i+1]-p[:,i-1])/2Δy
    end

    return u_geo,v_geo
end

function Q1(x::AbstractFloat,y::AbstractFloat,L=2)
    if ( 4L < x < 6L)
        return sin( pi /(2*L) * x) * exp(-1/4 * y^2)
    else
        return 0
    end
end

function Q2(x::AbstractFloat,y::AbstractFloat,L=2)
    if ( 4L < x < 6L )
        return sin( pi /(2*L) * x) * y * exp(-1/4 * y^2)
    else
        return 0
    end
end

function Q3(x::AbstractFloat,y::AbstractFloat,L=2)
    if ( 4L < x < 6L && y > 0)
        return sin( pi /(2*L) * x) * y * exp(-1/4 * y^2)
    else
        return 0
    end
end

using DrWatson
@quickactivate "numerical_gill" # <- project name
using LoopVectorization

function gill!(dx::Array, x::Array, constants)
    u=view(x,1,:,:)
    v=view(x,2,:,:)
    p=view(x,3,:,:)
    x_dim, y_dim, Δt, Δx, Δy, ys, Q, ϵ_u, ϵ_v, ϵ_p  = constants
    @turbo for j in 2:y_dim-1
        for i in 2:x_dim-1
            dx[1,i,j] = - ϵ_u * u[i,j] + 1/2 * ys[j] * v[i,j] - (p[i+1,j]-p[i-1,j])/2Δx
            dx[2,i,j] = - ϵ_v * v[i,j] - 1/2 * ys[j] * u[i,j] - (p[i,j+1]-p[i,j-1])/2Δy
            dx[3,i,j] = - ϵ_p * p[i,j] - (u[i+1,j]-u[i-1,j])/2Δx - (v[i,j+1]-v[i,j-1])/2Δy + Q[i,j]
        end
    end
end

function gill_EF!(dx::Array, x::Array, constants)
    # euler forward in space
    u=view(x,1,:,:)
    v=view(x,2,:,:)
    p=view(x,3,:,:)
    x_dim, y_dim, Δt, Δx, Δy, ys, Q, ϵ_u, ϵ_v, ϵ_p  = constants
    @turbo for j in 2:y_dim-1
        for i in 2:x_dim-1
            dx[1,i,j] = - ϵ_u * u[i,j] + 1/2 * ys[j] * v[i,j] - (p[i+1,j]-p[i,j])/Δx
            dx[2,i,j] = - ϵ_v * v[i,j] - 1/2 * ys[j] * u[i,j] - (p[i,j+1]-p[i,j])/Δy
            dx[3,i,j] = - ϵ_p * p[i,j] - (u[i+1,j]-u[i,j])/Δx - (v[i,j+1]-v[i,j])/Δy + Q[i,j]
        end
    end
end

function DCD(f,delta)
    return (-f[5]+8*f[4]-8*f[2]+f[1])/(12*delta)
end

function gill_DCD!(dx::Array, x::Array, constants)
    # double central differences in space
    # https://en.wikipedia.org/wiki/Five-point_stencil
    u=view(x,1,:,:)
    v=view(x,2,:,:)
    p=view(x,3,:,:)
    x_dim, y_dim, Δt, Δx, Δy, ys, Q, ϵ_u, ϵ_v, ϵ_p  = constants
    for j in 3:y_dim-2
        for i in 3:x_dim-2
            dx[1,i,j] = - ϵ_u * u[i,j] + 1/2 * ys[j] * v[i,j] - DCD(p[i-2:i+2,j],Δx)
            dx[2,i,j] = - ϵ_v * v[i,j] - 1/2 * ys[j] * u[i,j] - DCD(p[i,j-2:j+2],Δy)
            dx[3,i,j] = - ϵ_p * p[i,j] - DCD(u[i-2:i+2,j],Δx) - DCD(v[i,j-2:j+2],Δy) + Q[i,j]
        end
    end
end

function WTG!(dx::Array, x::Array, constants)
    # according to Bretherton and Sobel 2003
    u=view(x,1,:,:)
    v=view(x,2,:,:)
    p=view(x,3,:,:)
    x_dim, y_dim, Δt, Δx, Δy, ys, Q, ϵ_u, ϵ_v, ϵ_p  = constants
    @turbo for i in 2:x_dim-1
        for j in 2:y_dim-1
            dx[1,i,j] = - ϵ_u * u[i,j] + 1/2 * ys[j] * v[i,j] - (p[i+1,j]-p[i-1,j])/2Δx
            dx[2,i,j] = - ϵ_v * v[i,j] - 1/2 * ys[j] * u[i,j] - (p[i,j+1]-p[i,j-1])/2Δy
            dx[3,i,j] = - (u[i+1,j]-u[i-1,j])/2Δx - (v[i,j+1]-v[i,j-1])/2Δy + Q[i,j]
        end
    end

end


function tuyl!(dx::Array,x::Array, constants)
    # Advection is prone to be numerically unstable.
    Ro = 0.02
    kappa = 343.
    #Rop = 0.02
    #kappa = 10000.
    u=view(x,1,:,:)
    v=view(x,2,:,:)
    p=view(x,3,:,:)
    x_dim, y_dim, Δt, Δx, Δy, ys, Q, ϵ_u, ϵ_v, ϵ_p  = constants
    @turbo for i in 2:x_dim-1
        for j in 2:y_dim-1
            dx[1,i,j] = - ϵ_u * u[i,j] + ys[j] * v[i,j] - (p[i+1,j]-p[i-1,j])/2Δx - Ro * (u[i,j] * (u[i+1,j]-u[i-1,j])/2Δx + v[i,j] * (u[i,j+1]-u[i,j-1])/2Δy )
            dx[2,i,j] = - ϵ_v * v[i,j] - ys[j] * u[i,j] - (p[i,j+1]-p[i,j-1])/2Δy - Ro * (u[i,j] * (v[i+1,j]-v[i-1,j])/2Δx + v[i,j] * (v[i,j+1]-v[i,j-1])/2Δy )
            dx[3,i,j] = - ϵ_p * p[i,j] - 1/(kappa) * ( (u[i+1,j]-u[i-1,j])/2Δx + (v[i,j+1]-v[i,j-1])/2Δy ) - Ro * ( (u[i+1,j] * p[i+1,j] - u[i-1,j] * p[i-1,j])/2Δx + (v[i,j+1] * p[i,j+1] - v[i,j-1] * p[i,j-1])/2Δy ) + Q[i,j]
        end
    end

end

function tuyl_diff!(dx::Array,x::Array, constants)
    # Advection is prone to be numerically unstable.
    # here we also add diffusion to ensure stability
    # https://maths.ucd.ie/met/msc/fezzik/Numer-Met/Linear_Advection.pdf
    Ro = 0.02
    kappa = 343.
    diff_c = 1.
    #Rop = 0.02
    #kappa = 10000.
    u=view(x,1,:,:)
    v=view(x,2,:,:)
    p=view(x,3,:,:)
    x_dim, y_dim, Δt, Δx, Δy, ys, Q, ϵ_u, ϵ_v, ϵ_p  = constants
    @turbo for i in 2:x_dim-1
        for j in 2:y_dim-1
            dx[1,i,j] = - ϵ_u * u[i,j] + ys[j] * v[i,j] - (p[i+1,j]-p[i-1,j])/2Δx - Ro * (u[i,j] * (u[i+1,j]-u[i-1,j])/2Δx + v[i,j] * (u[i,j+1]-u[i,j-1])/2Δy ) + diff_c * ((u[i+1,j]-2*u[i,j]+u[i-1,j]) + (u[i,j+1]-2*u[i,j]+u[i,j-1]))
            dx[2,i,j] = - ϵ_v * v[i,j] - ys[j] * u[i,j] - (p[i,j+1]-p[i,j-1])/2Δy - Ro * (u[i,j] * (v[i+1,j]-v[i-1,j])/2Δx + v[i,j] * (v[i,j+1]-v[i,j-1])/2Δy ) + diff_c * ((v[i+1,j]-2*v[i,j]+v[i-1,j]) + (v[i,j+1]-2*v[i,j]+v[i,j-1]))
            dx[3,i,j] = - ϵ_p * p[i,j] - 1/(kappa) * ( (u[i+1,j]-u[i-1,j])/2Δx + (v[i,j+1]-v[i,j-1])/2Δy ) - Ro * ( (u[i+1,j] * p[i+1,j] - u[i-1,j] * p[i-1,j])/2Δx + (v[i,j+1] * p[i,j+1] - v[i,j-1] * p[i,j-1])/2Δy ) + Q[i,j] + diff_c * ((p[i+1,j] - 2*p[i,j] + p[i-1,j]) + (p[i,j+1] - 2*p[i,j] + p[i,j-1]))
        end
    end

end

function gill_diff!(dx::Array, x::Array, constants)
    u=view(x,1,:,:)
    v=view(x,2,:,:)
    p=view(x,3,:,:)
    x_dim, y_dim, Δt, Δx, Δy, ys, Q, ϵ_u, ϵ_v, ϵ_p  = constants
    diff_c = 0.1
    @turbo for j in 2:y_dim-1
        for i in 2:x_dim-1
            dx[1,i,j] = - ϵ_u * u[i,j] + 1/2 * ys[j] * v[i,j] - (p[i+1,j]-p[i-1,j])/2Δx + diff_c * ((u[i+1,j]-2*u[i,j]+u[i-1,j]) + (u[i,j+1]-2*u[i,j]+u[i,j-1]))
            dx[2,i,j] = - ϵ_v * v[i,j] - 1/2 * ys[j] * u[i,j] - (p[i,j+1]-p[i,j-1])/2Δy + diff_c * ((v[i+1,j]-2*v[i,j]+v[i-1,j]) + (v[i,j+1]-2*v[i,j]+v[i,j-1]))
            dx[3,i,j] = - ϵ_p * p[i,j] - (u[i+1,j]-u[i-1,j])/2Δx - (v[i,j+1]-v[i,j-1])/2Δy + Q[i,j] + diff_c * ((p[i+1,j] - 2*p[i,j] + p[i-1,j]) + (p[i,j+1] - 2*p[i,j] + p[i,j-1]))
        end
    end
end

function gill_EMF!(dx::Array, x::Array, constants)
    # Sobel and Schneider 2009
    u=view(x,1,:,:)
    v=view(x,2,:,:)
    p=view(x,3,:,:)
    x_dim, y_dim, Δt, Δx, Δy, ys, Q, ϵ_u, ϵ_v, ϵ_p  = constants
    vd = 0.01 #EMF parameter. unstable for 0.05, stable for 0.01
    diff_c = 0.05
    @turbo for j in 2:y_dim-1
        for i in 2:x_dim-1
            dx[1,i,j] = - ϵ_u * u[i,j] + 1/2 * ys[j] * v[i,j] - (p[i+1,j]-p[i-1,j])/2Δx - vd * heaviside(u[i,j]) * sign(ys[j]) * (u[i,j+1]-u[i,j-1])/2Δy + diff_c * ((u[i+1,j]-2*u[i,j]+u[i-1,j]) + (u[i,j+1]-2*u[i,j]+u[i,j-1]))
            dx[2,i,j] = - ϵ_v * v[i,j] - 1/2 * ys[j] * u[i,j] - (p[i,j+1]-p[i,j-1])/2Δy + diff_c * ((v[i+1,j]-2*v[i,j]+v[i-1,j]) + (v[i,j+1]-2*v[i,j]+v[i,j-1]))
            dx[3,i,j] = - ϵ_p * p[i,j] - (u[i+1,j]-u[i-1,j])/2Δx - (v[i,j+1]-v[i,j-1])/2Δy + Q[i,j] + diff_c * ((p[i+1,j] - 2*p[i,j] + p[i-1,j]) + (p[i,j+1] - 2*p[i,j] + p[i,j-1]))
        end
    end
end

function heaviside(a)
   0.5 * (sign(a) + 1)
end


function wu_2015(x::Array, constants::Vector)
    # this does not really make sense
    u=x[1]
    v=x[2]
    p=x[3]
    fu = zeros(size(u))
    fv = zeros(size(v))
    fp = zeros(size(p))
    x_dim, y_dim, Δt, Δx, Δy, ys, dQdz, ϵ = constants
    for i in 2:x_dim-1
        for j in 2:y_dim-1
            fu[i,j] = - ϵ * u[i,j] + 1/2 * ys[j] * v[i,j] - (p[i+1,j]-p[i-1,j])/2Δx
            fv[i,j] = - ϵ * v[i,j] - 1/2 * ys[j] * u[i,j] - (p[i,j+1]-p[i,j-1])/2Δy
            fp[i,j] = - ϵ * p[i,j] + v[i,j] * ys[j] + ((2*ϵ)/ys[j]) * ( (v[i+1,j]-v[i-1,j])/2Δx - (u[i,j+1]-u[i,j-1])/2Δy ) - dQdz[i,j]
        end
    end

    return [fu,fv,fp]
end

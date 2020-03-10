using LyceumMuJoCoViz: EngineMode, UIState, PhysicsState, default_windowsize, PassiveDynamics, Engine, onkey, GLFW, MOD_CONTROL

function LyceumMuJoCoViz.visualize(
    model::Union{AbstractString,MJSim,AbstractMuJoCoEnvironment},
    modes::EngineMode...;
    windowsize::NTuple{2,Integer} = default_windowsize()
)
    model isa AbstractString && (model = MJSim(model))
    reset!(model)

    modes = EngineMode[PassiveDynamics(), modes...]
    LyceumMuJoCoViz.run(Engine(windowsize, model, Tuple(modes)))
    return
end

mutable struct HEBIManualChop{F} <: EngineMode
    controller::F
    realtimefactor::Float64
end
HEBIManualChop(controller) = HEBIManualChop(controller, 1.0)

function LyceumMuJoCoViz.setup!(ui::UIState, p::PhysicsState, x::HEBIManualChop)
    dt = @elapsed x.controller(p.model)
    x.realtimefactor = timestep(p.model) / dt
    return ui
end

function LyceumMuJoCoViz.teardown!(ui::UIState, p::PhysicsState, x::HEBIManualChop)
    zerofullctrl!(getsim(p.model))
    return ui
end

function LyceumMuJoCoViz.forwardstep!(p::PhysicsState, x::HEBIManualChop)
    dt = @elapsed x.controller(p.model)
    x.realtimefactor = timestep(p.model) / dt
    return LyceumMuJoCoViz.forwardstep!(p)
end

function LyceumMuJoCoViz.modeinfo(io1, io2, ui::UIState, p::PhysicsState, x::HEBIManualChop)
    println(io1, "Realtime Factor")
    @printf io2 "%.2fx\n" x.realtimefactor
    return nothing
end

function LyceumMuJoCoViz.handlers(ui::UIState, p::PhysicsState, m::HEBIManualChop)
    return let ui=ui, p=p, m=m
        [
            onkey(GLFW.KEY_W, MOD_CONTROL, what = "Open Chopsticks") do s, ev
                if LyceumMuJoCoViz.ispress_or_repeat(ev.action)
                    a = getaction(p.model)
                    a[end] = clamp(a[end] + 0.01, -1, 1)
                    setaction!(p.model, a)
                end
            end,

            onkey(GLFW.KEY_S, MOD_CONTROL, what = "Close Chopsticks") do s, ev
                if LyceumMuJoCoViz.ispress_or_repeat(ev.action)
                    a = getaction(p.model)
                    a[end] = clamp(a[end] - 0.01, -1, 1)
                    setaction!(p.model, a)
                end
            end,
        ]
    end
end

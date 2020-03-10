
using StaticArrays
function SArray{S,T,N,L}(A::AbstractArray{<:Any,N}, I::Vararg{<:Any, N}) where {S,T,N,L}
    SArray{S,T,N,L}(view(A, I...)...)
end
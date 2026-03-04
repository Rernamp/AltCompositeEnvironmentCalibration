#let r = $arrow(r)$

$#r$ - const ray
#let w = $w$
#let u = $U$
#let q = $q$

#let q_ex = $[#w, #u]$
$#q = #q_ex$ - rotation quaternion

#let R = $arrow(R)$
#let R_ex = $#w^2 * #r + 2 dot (#w * (#u times #r) + #u * (#u dot #r)) - (#u dot #u) * #r$

$#R = #R_ex$ - rotated vector

#let p = $arrow(p)$
#let m = $arrow(m)$
#let pm = $arrow("pm")$
#let pm_ex = $#m - #p$

$#pm = #pm_ex$ - position to marker vector

#let cost_func_f = $f$
#let cost_func_from_cos = $cos(alpha)$

$#cost_func_f = #cost_func_from_cos$

#let cost_func_norms_part_f = $f_"norm_part"$
#let cost_func_norms_part = $(norm(#R) * norm(#pm))$
#let cost_func_dot_part_f = $f_"dot_part"$
#let cost_func_dot_part = $(#R dot #pm)$

$#cost_func_dot_part_f = #cost_func_dot_part$

$#cost_func_norms_part_f = #cost_func_norms_part$

#let cost_func_ex = $#cost_func_dot_part_f / #cost_func_norms_part_f$

$#cost_func_from_cos = #cost_func_ex$

$#cost_func_f = #cost_func_ex = (#cost_func_dot_part) / (#cost_func_norms_part)$

#let nabla_m = $nabla_#m$

Вспомогательная формула:

$nabla(U/V) = (V * nabla U - U * nabla V)/(V^2)$ 

#let cost_func_norms_part_nabla_m = $#nabla_m #cost_func_norms_part_f$
#let cost_func_dot_part_nabla_m = $#nabla_m #cost_func_dot_part_f$

$#nabla_m #cost_func_f = #nabla_m (#cost_func_ex) = (#cost_func_norms_part_f * (#cost_func_dot_part_nabla_m) - #cost_func_dot_part_f * (#cost_func_norms_part_nabla_m)) / (#cost_func_norms_part_f^2)$

Вспомогательная формула:

$nabla (arrow(A) dot arrow(B)) = (arrow(A) dot nabla) * arrow(B) + (arrow(B) dot nabla) * arrow(A) + arrow(A) times (nabla times arrow(B)) + arrow(B) times (nabla times arrow(A))$

$#cost_func_dot_part_nabla_m = #nabla_m (#cost_func_dot_part)$



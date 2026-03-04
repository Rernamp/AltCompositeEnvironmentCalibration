#let r = $arrow(r)$

$#r$ - const ray
#let w = $w$
#let u = $arrow(u)$
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

#let cost_func = $f$
#let cost_func_ex = $cos(alpha)$

$#cost_func = #cost_func_ex$

$#cost_func = (#R dot #pm) / (norm(#R) * norm(#pm))$
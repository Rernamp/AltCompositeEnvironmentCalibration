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

#let nabla_from_dot_product(by_param, vector_A, vector_B) = {
  let nabla_by_param = $nabla_#by_param$
  let dot_nabla_to_B = $(#vector_A dot #nabla_by_param) * #vector_B$
  let dot_nabla_to_A = $(#vector_B dot #nabla_by_param) * #vector_A$
  let cross_nabla_to_B = $#vector_A times (#nabla_by_param times #vector_B)$
  let cross_nabla_to_A = $#vector_B times (#nabla_by_param times #vector_A)$
  let nabla_dot_prod_result = $#dot_nabla_to_B + #dot_nabla_to_A + #cross_nabla_to_B + #cross_nabla_to_A$
  let nabla_dot_prod_f = $#nabla_by_param (#vector_A dot #vector_B)$

  return (nabla_dot_prod_result, nabla_dot_prod_f, dot_nabla_to_B, dot_nabla_to_A, cross_nabla_to_B, cross_nabla_to_A)
}

#let (nabla_example_result_ex, nabla_example_f, ..) = nabla_from_dot_product("u", $arrow(A)$, $arrow(B)$)

$#nabla_example_f = #nabla_example_result_ex$

#let (const_func_dot_part_nabla_by_m_ex, const_func_dot_part_nabla_by_m_f,
const_func_dot_part_nabla_by_m_dot_to_pm, const_func_dot_part_nabla_by_m_dot_to_R, ..)   = nabla_from_dot_product(m, R, pm)

$#cost_func_dot_part_nabla_m = #nabla_m (#cost_func_dot_part) = #const_func_dot_part_nabla_by_m_ex$

$#const_func_dot_part_nabla_by_m_dot_to_pm = (#R dot #nabla_m) * (#pm_ex)$

$(#R dot #nabla_m) * #p = arrow(0)$

#let dot_nabla_to_vector(to_param, left_vector, right_vector) = {
  let nabla_by_param = $nabla_#to_param$
  let result_f = $(#left_vector dot #nabla_by_param) * #right_vector$
  let result_ex = $#left_vector$
  return (result_f, result_ex)
}

#let (dot_nabla_by_m_to_m_f, dot_nabla_by_m_to_m_ex) = dot_nabla_to_vector(m, R, m)

$#dot_nabla_by_m_to_m_f = #dot_nabla_by_m_to_m_ex$

$#const_func_dot_part_nabla_by_m_dot_to_pm = #dot_nabla_by_m_to_m_ex$

#let const_func_dot_part_nabla_by_m_dot_to_R_ex = $arrow(0)$
$#const_func_dot_part_nabla_by_m_dot_to_R = #const_func_dot_part_nabla_by_m_dot_to_R_ex$

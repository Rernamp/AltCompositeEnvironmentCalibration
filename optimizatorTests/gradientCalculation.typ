#let r = $arrow(r)$

$#r$ - const ray
#let w = $w$
#let u = $arrow(U)$
#let q = $q$

#let q_ex = $[#w, #u]$
$#q = #q_ex$ - rotation quaternion

#let R = $arrow(R)$
#let R_ex = $#w^2 * #r + 2 * (#w * (#u times #r) + #u * (#u dot #r)) - (#u dot #u) * #r$

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
#let R_norm_ex = $norm(#R)$
#let pm_norm_ex = $norm(#pm)$
#let cost_func_norms_part_ex = $(#R_norm_ex * #pm_norm_ex)$
#let cost_func_dot_part_f = $f_"dot_part"$
#let cost_func_dot_part_ex = $(#R dot #pm)$

$#cost_func_dot_part_f = #cost_func_dot_part_ex$

$#cost_func_norms_part_f = #cost_func_norms_part_ex$

#let cost_func_ex = $#cost_func_dot_part_f / #cost_func_norms_part_f$

$#cost_func_from_cos = #cost_func_ex$

$#cost_func_f = #cost_func_ex = (#cost_func_dot_part_ex) / (#cost_func_norms_part_ex)$

#let nabla_m = $nabla_#m$

#block(
  inset: 1em,
  stroke: 0.5pt + gray,
  $nabla(U/V) = (V * nabla U - U * nabla V)/(V^2)$,
)


#let cost_func_norms_part_nabla_m = $#nabla_m #cost_func_norms_part_f$
#let cost_func_dot_part_nabla_m = $#nabla_m #cost_func_dot_part_f$

$#nabla_m #cost_func_f = #nabla_m (#cost_func_ex) = (#cost_func_norms_part_f * (#cost_func_dot_part_nabla_m) - #cost_func_dot_part_f * (#cost_func_norms_part_nabla_m)) / (#cost_func_norms_part_f^2)$

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

#block(
  inset: 1em,
  stroke: 0.5pt + gray,
  $#nabla_example_f = #nabla_example_result_ex$,
)

#let (
  const_func_dot_part_nabla_by_m_ex,
  const_func_dot_part_nabla_by_m_f,
  const_func_dot_part_nabla_by_m_dot_to_pm,
  const_func_dot_part_nabla_by_m_dot_to_R,
  const_func_dot_part_nabla_by_m_cross_to_pm,
  const_func_dot_part_nabla_by_m_cross_to_R,
  ..,
) = nabla_from_dot_product(m, R, pm)

#block(
  inset: 1em,
  stroke: 0.5pt + gray,
  $#cost_func_dot_part_nabla_m = #nabla_m (#cost_func_dot_part_ex) = #const_func_dot_part_nabla_by_m_ex$,
)

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

#block(
  inset: 1em,
  stroke: 0.5pt + gray,
  $#const_func_dot_part_nabla_by_m_dot_to_pm = #dot_nabla_by_m_to_m_ex$,
)

#let const_func_dot_part_nabla_by_m_dot_to_R_ex = $arrow(0)$
#block(
  inset: 1em,
  stroke: 0.5pt + gray,
  $#const_func_dot_part_nabla_by_m_dot_to_R = #const_func_dot_part_nabla_by_m_dot_to_R_ex$,
)

#let const_func_dot_part_nabla_by_m_cross_to_pm_ex = $arrow(0)$
#let nabla_m_cross_pm = $(#nabla_m times (#pm_ex))$
$#const_func_dot_part_nabla_by_m_cross_to_pm = #R times (#nabla_m_cross_pm)$

#let nabla_m_cross_m = $(#nabla_m times #m)$
#let nabla_m_cross_p = $(#nabla_m times #p)$

$#nabla_m_cross_pm = #nabla_m_cross_m - #nabla_m_cross_p$

#let nabla_m_cross_m_ex = $arrow(0)$
#let nabla_m_cross_p_ex = $arrow(0)$

$#nabla_m_cross_m = #nabla_m_cross_m_ex$

$#nabla_m_cross_p = #nabla_m_cross_p_ex$

#let nabla_m_cross_pm_ex = $arrow(0)$

#block(
  inset: 1em,
  stroke: 0.5pt + gray,
  $#nabla_m_cross_pm = #nabla_m_cross_pm_ex$,
)

#let const_func_dot_part_nabla_by_m_cross_to_R_ex = $arrow(0)$
#block(
  inset: 1em,
  stroke: 0.5pt + gray,
  $#const_func_dot_part_nabla_by_m_cross_to_R = #const_func_dot_part_nabla_by_m_cross_to_R_ex$,
)

#let cost_func_dot_part_nabla_m_ex = $#dot_nabla_by_m_to_m_ex$

#block(
  inset: 1em,
  stroke: 0.5pt + gray,
  $#cost_func_dot_part_nabla_m = #cost_func_dot_part_nabla_m_ex$,
)

#let pm_norm_nabla_by_m_f = $#nabla_m #pm_norm_ex$

#let cost_func_norms_part_nabla_m_ex = $#nabla_m #cost_func_norms_part_ex$
$#cost_func_norms_part_nabla_m = #cost_func_norms_part_nabla_m_ex = #R_norm_ex * #pm_norm_nabla_by_m_f$

$#pm_norm_nabla_by_m_f = #nabla_m norm(#pm_ex)$

#let nabla_to_norm_vector(by_param, Vector) = {
    let nabla_by_param = $nabla_#by_param$
    let v = $#nabla_by_param norm(Vector)$
    let ex = $(#nabla_by_param (Vector dot Vector))/(2 norm(Vector))$
    return (v, ex)
}

#{
    let (view, ex) = nabla_to_norm_vector("u", $arrow(A)$)
    block(
        inset: 1em,
        stroke: 0.5pt + gray,        
        $view = ex \
        nabla_arrow(u)(arrow(A) dot arrow(A)) = 2 ((arrow(A) dot nabla_arrow(u)) * arrow(A) + arrow(A) times (nabla_arrow(u) times arrow(A)))
        $
    )
}


#let nabla_m_to_pm = $#nabla_m norm(#pm)$

#{    
    let pm_dot_nambla_m_to_pm = $((#pm) dot #nabla_m) * (#pm)$
    let pm_times_nabla_m_to_pm = $(#pm) times (#nabla_m times (#pm))$
    $#nabla_m_to_pm = #nabla_m ((#pm) dot (#pm)) = 2 (#pm_dot_nambla_m_to_pm + #pm_times_nabla_m_to_pm$

    parbreak()

    $#pm_dot_nambla_m_to_pm = #pm$

    parbreak()

    $#pm_times_nabla_m_to_pm = arrow(0)$
}

#let nabla_m_to_pm_ex = $#pm$


$#nabla_m_to_pm = #nabla_m_to_pm_ex$


#let cost_func_norms_part_nabla_m_ex = $#R_norm_ex * #nabla_m_to_pm_ex$
#block(
    inset: 1em,
    stroke: 0.5pt + gray,
    $#cost_func_norms_part_nabla_m = #cost_func_norms_part_nabla_m_ex$
)

$#nabla_m #cost_func_f = (#cost_func_norms_part_ex * (#cost_func_dot_part_nabla_m_ex) - #cost_func_dot_part_ex * (#cost_func_norms_part_nabla_m_ex)) / (#cost_func_norms_part_ex^2)$

#block(
    inset: 1em,
    stroke: 0.5pt + gray,
    $#nabla_m #cost_func_f = (#R_norm_ex) / (#cost_func_norms_part_ex) - (#cost_func_dot_part_ex * (#cost_func_norms_part_nabla_m_ex)) / (#cost_func_norms_part_ex^2)$
)

Аналогично:

#let cost_func_by_nabla_m_ex = $(#R_norm_ex) / (#cost_func_norms_part_ex) - (#cost_func_dot_part_ex * (#cost_func_norms_part_nabla_m_ex)) / (#cost_func_norms_part_ex^2)$

#block(
    inset: 1em,
    stroke: 0.5pt + gray,
    $#nabla_m #cost_func_f = #cost_func_by_nabla_m_ex$
)

#let nabla_p = $nabla_#p$

#block(
    inset: 1em,
    stroke: 0.5pt + gray,
    $#nabla_p #cost_func_f = -(#nabla_m #cost_func_f)$
)

#let deriv_w = $d / (d #w)$

$#deriv_w #cost_func_norms_part_f = #deriv_w #cost_func_norms_part_ex$

$#deriv_w #cost_func_norms_part_ex = #pm_norm_ex * #deriv_w #R_norm_ex$

#let deriv_R_by_w_ex_without_scale = $(#w * #r + (#u times #r))$
#let deriv_R_by_w_ex = $2 * #deriv_R_by_w_ex_without_scale$
#{
    let const_values = $1 / (2 #R_norm_ex)$
    let deriv_part = $#deriv_w (#R dot #R)$
    let deriv_R_by_w = $#deriv_w #R$
    $#deriv_w #R_norm_ex = #const_values #deriv_part \
    #deriv_part = (#deriv_w #R dot #R) + (#R dot #deriv_w #R) = 2 (#R dot #deriv_w #R) \
    #deriv_R_by_w = #deriv_w (#R_ex) \
    #deriv_R_by_w = #deriv_R_by_w_ex
    $
}

#let cost_func_norms_part_f_deriv_by_w = $(2 #pm_norm_ex * (#R dot #deriv_R_by_w_ex_without_scale)) / (#R_norm_ex)$

$#deriv_w #cost_func_norms_part_f = (#pm_norm_ex * (#R dot (#deriv_R_by_w_ex))) / (#R_norm_ex)$


#block(
    inset: 1em,
    stroke: 0.5pt + gray,
$#deriv_w #cost_func_norms_part_f = #cost_func_norms_part_f_deriv_by_w$
)
#let cost_func_dot_part_ex_deriv_by_w = $2 (#deriv_R_by_w_ex_without_scale dot #pm)$

$#deriv_w #cost_func_dot_part_f = #deriv_w #cost_func_dot_part_ex = (#deriv_w #R dot #pm)$

#block(
    inset: 1em,
    stroke: 0.5pt + gray,
    $#deriv_w #cost_func_dot_part_f = #cost_func_dot_part_ex_deriv_by_w$
)

#block(
    inset: 1em,
    stroke: 0.5pt + gray,
$#deriv_w #cost_func_f = (#cost_func_norms_part_f * #deriv_w #cost_func_dot_part_f - #cost_func_dot_part_f * #deriv_w #cost_func_norms_part_f) / (#cost_func_norms_part_f^2)$
)


#let nabla_u = $nabla_#u$

$#nabla_u #cost_func_dot_part_f = #nabla_u #cost_func_dot_part_ex$

#let (nabla_cost_f_by_u_ex, nabla_cost_f_by_u_f, ..) = nabla_from_dot_product($#u$, $#R$, $#pm$)

$#nabla_cost_f_by_u_f = #nabla_cost_f_by_u_ex$
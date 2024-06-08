/*
 * File: advs.c
 *
 */

static double powd_snf(double u0, double u1)
{
  double y;
  double d0;
  double d1;
  if (isnan(u0) || isnan(u1)) {
    y = NAN;
  } else {
    d0 = fabs(u0);
    d1 = fabs(u1);

    if (isinf(u1)) {
      if (d0 == 1.0) {
        y = NAN;
      } else if (d0 > 1.0) {
        if (u1 > 0.0) {
          y = INFINITY;
        } else {
          y = 0.0;
        }
      } else if (u1 > 0.0) {
        y = 0.0;
      } else {
        y = INFINITY;
      }
    } else if (d1 == 0.0) {
      y = 1.0;
    } else if (d1 == 1.0) {
      if (u1 > 0.0) {
        y = u0;
      } else {
        y = 1.0 / u0;
      }
    } else if (u1 == 2.0) {
      y = u0 * u0;
    } else if ((u1 == 0.5) && (u0 >= 0.0)) {
      y = sqrt(u0);
    } else if ((u0 < 0.0) && (u1 > floor(u1))) {
      y = NAN;
    } else {
      y = pow(u0, u1);
    }
  }

  return y;
}

/*
 * These are the strains in a uniform elastic half space due to slip
 *  on an angular dislocation.  They were calculated by symbolically
 *  differentiating the expressions for the displacements (Comninou and
 *  Dunders, 1975, with typos noted by Thomas 1993) then combining the
 *  elements of the displacement gradient tensor to form the strain tensor.
 * Arguments    : double b_y1
 *                double y2
 *                double y3
 *                double a
 *                double b
 *                double nu
 *                double B1
 *                double B2
 *                double B3
 *                double *e11
 *                double *e22
 *                double *e33
 *                double *e12
 *                double *e13
 *                double *e23
 * Return Type  : void
 */
void advs(double b_y1, double y2, double y3, double a, double b, double nu,
          double B1, double B2, double B3, double *e11, double *e22, double *e33,
          double *e12, double *e13, double *e23)
{
  double b_a;
  double c_a;
  double d_a;
  double x;
  double e_a;
  double f_a;
  double g_a;
  double h_a;
  double i_a;
  double j_a;
  double k_a;
  double b_x;
  double l_a;
  double m_a;
  double n_a;
  double o_a;
  double p_a;
  double q_a;
  double r_a;
  double s_a;
  double t_a;
  double u_a;
  double v_a;
  double w_a;
  double x_a;
  double y_a;
  double ab_a;
  double bb_a;
  double cb_a;
  double db_a;
  double eb_a;
  double fb_a;
  double gb_a;
  double hb_a;
  double ib_a;
  double jb_a;
  double kb_a;
  double c_x;
  double lb_a;
  double d_x;
  double mb_a;
  double nb_a;
  double ob_a;
  double pb_a;
  double qb_a;
  double rb_a;
  double sb_a;
  double tb_a;
  double ub_a;
  double vb_a;
  double wb_a;
  double xb_a;
  double yb_a;
  double ac_a;
  double bc_a;
  double cc_a;
  double dc_a;
  double ec_a;
  double fc_a;
  double gc_a;
  double hc_a;
  double ic_a;
  double jc_a;
  double kc_a;
  double lc_a;
  double mc_a;
  double nc_a;
  double oc_a;
  double pc_a;
  double qc_a;
  double rc_a;
  double sc_a;
  double tc_a;
  double uc_a;
  double vc_a;
  double wc_a;
  double xc_a;
  double yc_a;
  double ad_a;
  double bd_a;
  double cd_a;
  double dd_a;
  double ed_a;
  double fd_a;
  double gd_a;
  double hd_a;
  double id_a;
  double jd_a;
  double kd_a;
  double ld_a;
  double md_a;
  double nd_a;
  double od_a;
  double pd_a;
  double qd_a;
  double rd_a;
  double sd_a;
  double td_a;
  double ud_a;
  double vd_a;
  double wd_a;
  double xd_a;
  double yd_a;
  double ae_a;
  double be_a;
  double ce_a;
  double de_a;
  double ee_a;
  double fe_a;
  double ge_a;
  double he_a;
  double ie_a;
  double je_a;
  double ke_a;
  double le_a;
  double me_a;
  double ne_a;
  double oe_a;
  double pe_a;
  double qe_a;
  double re_a;
  double se_a;
  double te_a;
  double ue_a;
  double ve_a;
  double we_a;
  double xe_a;
  double ye_a;
  double af_a;
  double bf_a;
  double cf_a;
  double df_a;
  double ef_a;
  double ff_a;
  double gf_a;
  double e_x;
  double hf_a;
  double if_a;
  double f_x;
  double jf_a;
  double kf_a;
  double lf_a;
  double mf_a;
  double nf_a;
  double of_a;
  double pf_a;
  double qf_a;
  double rf_a;
  double sf_a;
  double tf_a;
  double uf_a;
  double vf_a;
  double wf_a;
  double xf_a;
  double yf_a;
  double ag_a;
  double bg_a;
  double cg_a;
  double dg_a;
  double eg_a;
  double fg_a;
  double g_x;
  double gg_a;
  double hg_a;
  double ig_a;
  double jg_a;
  double kg_a;
  double lg_a;
  double mg_a;
  double ng_a;
  double og_a;
  double pg_a;
  double qg_a;
  double rg_a;
  double sg_a;
  double tg_a;
  double ug_a;
  double vg_a;
  double wg_a;
  double xg_a;
  double yg_a;
  double ah_a;
  double bh_a;
  double ch_a;
  double dh_a;
  double eh_a;
  double fh_a;
  double gh_a;
  double hh_a;
  double ih_a;
  double jh_a;
  double kh_a;
  double lh_a;
  double mh_a;
  double nh_a;
  double oh_a;
  double ph_a;
  double qh_a;
  double rh_a;
  double sh_a;
  double th_a;
  double uh_a;
  double vh_a;
  double wh_a;
  double xh_a;
  double yh_a;
  double ai_a;
  double bi_a;
  double ci_a;
  double di_a;
  double ei_a;
  double fi_a;
  double gi_a;
  double hi_a;
  double ii_a;
  double ji_a;
  double ki_a;
  double li_a;
  double mi_a;
  double ni_a;
  double oi_a;
  double pi_a;
  double qi_a;
  double ri_a;
  double si_a;
  double ti_a;
  double ui_a;
  double vi_a;
  double wi_a;
  double xi_a;
  double yi_a;
  double aj_a;
  double bj_a;
  double cj_a;
  double dj_a;
  double ej_a;
  double fj_a;
  double gj_a;
  double hj_a;
  double ij_a;
  double jj_a;
  double kj_a;
  double lj_a;
  double mj_a;
  double nj_a;
  double oj_a;
  double pj_a;
  double qj_a;
  double rj_a;
  double sj_a;
  double tj_a;
  double uj_a;
  double vj_a;
  double wj_a;
  double xj_a;
  double yj_a;
  double ak_a;
  double bk_a;
  double ck_a;
  double dk_a;
  double ek_a;
  double fk_a;
  double gk_a;
  double hk_a;
  double ik_a;
  double jk_a;
  double kk_a;
  double lk_a;
  double mk_a;
  double nk_a;
  double ok_a;
  double pk_a;
  double qk_a;
  double rk_a;
  double sk_a;
  double tk_a;
  double uk_a;
  double vk_a;
  double h_x;
  double i_x;
  double j_x;
  double k_x;
  double wk_a;
  double xk_a;
  double yk_a;
  double al_a;
  double bl_a;
  double cl_a;
  double dl_a;
  double el_a;
  double fl_a;
  double gl_a;
  double hl_a;
  double il_a;
  double jl_a;
  double kl_a;
  double ll_a;
  double ml_a;
  double nl_a;
  double ol_a;
  double pl_a;
  double ql_a;
  double rl_a;
  double sl_a;
  double tl_a;
  double ul_a;
  double vl_a;
  double wl_a;
  double xl_a;
  double yl_a;
  double am_a;
  double bm_a;
  double cm_a;
  double dm_a;
  double em_a;
  double fm_a;
  double gm_a;
  double hm_a;
  double im_a;
  double jm_a;
  double km_a;
  double lm_a;
  double mm_a;
  double nm_a;
  double om_a;
  double pm_a;
  double qm_a;
  double rm_a;
  double sm_a;
  double tm_a;
  double um_a;
  double vm_a;
  double wm_a;
  double xm_a;
  double ym_a;
  double an_a;
  double bn_a;
  double cn_a;
  double dn_a;
  double en_a;
  double fn_a;
  double gn_a;
  double hn_a;
  double in_a;
  double jn_a;
  double kn_a;
  double ln_a;
  double mn_a;
  double nn_a;
  double on_a;
  double pn_a;
  double qn_a;
  double rn_a;
  double sn_a;
  double tn_a;
  double un_a;
  double vn_a;
  double wn_a;
  double xn_a;
  double yn_a;
  double ao_a;
  double bo_a;
  double co_a;
  double do_a;
  double eo_a;
  double fo_a;
  double go_a;
  double ho_a;
  double l_x;
  double io_a;
  double jo_a;
  double ko_a;
  double lo_a;
  double mo_a;
  double no_a;
  double oo_a;
  double po_a;
  double qo_a;
  double ro_a;
  double so_a;
  double to_a;
  double uo_a;
  double m_x;
  double vo_a;
  double wo_a;
  double xo_a;
  double yo_a;
  double ap_a;
  double bp_a;
  double cp_a;
  double n_x;
  double dp_a;
  double ep_a;
  double fp_a;
  double gp_a;
  double hp_a;
  double ip_a;
  double jp_a;
  double kp_a;
  double lp_a;
  double mp_a;
  double np_a;
  double op_a;
  double pp_a;
  double qp_a;
  double rp_a;
  double sp_a;
  double tp_a;
  double up_a;
  double vp_a;
  double wp_a;
  double xp_a;
  double yp_a;
  double o_x;
  double aq_a;
  double p_x;
  double bq_a;
  double cq_a;
  double dq_a;
  double eq_a;
  double fq_a;
  double gq_a;
  double hq_a;
  double iq_a;
  double jq_a;
  double kq_a;
  double lq_a;
  double mq_a;
  double nq_a;
  double oq_a;
  double pq_a;
  double qq_a;
  double rq_a;
  double sq_a;
  double tq_a;
  double uq_a;
  double vq_a;
  double wq_a;
  double xq_a;
  double yq_a;
  double ar_a;
  double br_a;
  double cr_a;
  double dr_a;
  double er_a;
  double fr_a;
  double gr_a;
  double hr_a;
  double ir_a;
  double jr_a;
  double kr_a;
  double lr_a;
  double mr_a;
  double nr_a;
  double or_a;
  double pr_a;
  double qr_a;
  double rr_a;
  double sr_a;
  double tr_a;
  double ur_a;
  double vr_a;
  double wr_a;
  double xr_a;
  double yr_a;
  double as_a;
  double bs_a;
  double cs_a;
  double ds_a;
  double es_a;
  double fs_a;
  double gs_a;
  double hs_a;
  double is_a;
  double js_a;
  double ks_a;
  double ls_a;
  double ms_a;
  double ns_a;
  double os_a;
  double ps_a;
  double qs_a;
  double rs_a;
  double ss_a;
  double ts_a;
  double us_a;
  double vs_a;
  double ws_a;
  double xs_a;
  double ys_a;
  double at_a;
  double bt_a;
  double ct_a;
  double dt_a;
  double et_a;
  double ft_a;
  double gt_a;
  double ht_a;
  double it_a;
  double jt_a;
  double kt_a;
  double lt_a;
  double mt_a;
  double nt_a;
  double ot_a;
  double pt_a;
  double qt_a;
  double rt_a;
  double st_a;
  double tt_a;
  double ut_a;
  double vt_a;
  double wt_a;
  double xt_a;
  double yt_a;
  double au_a;
  double bu_a;
  double cu_a;
  double du_a;
  double eu_a;
  double fu_a;
  double gu_a;
  double hu_a;
  double iu_a;
  double ju_a;
  double ku_a;
  double lu_a;
  double mu_a;
  double nu_a;
  double ou_a;
  double pu_a;
  double qu_a;
  double ru_a;
  double su_a;
  double tu_a;
  double uu_a;
  double vu_a;
  double wu_a;
  double xu_a;
  double yu_a;
  double av_a;
  double bv_a;
  double cv_a;
  double dv_a;
  double ev_a;
  double fv_a;
  double gv_a;
  double hv_a;
  double iv_a;
  double jv_a;
  double kv_a;
  double lv_a;
  double mv_a;
  double nv_a;
  double ov_a;
  b_a = b_y1 * cos(b) - y3 * sin(b);
  c_a = b_y1 * cos(b) - y3 * sin(b);
  d_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  x = sin(b);
  e_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  f_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  g_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  h_a = y3 + 2.0 * a;
  i_a = y3 + 2.0 * a;
  j_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  k_a = y3 + 2.0 * a;
  b_x = sin(b);
  l_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  m_a = y3 + 2.0 * a;
  n_a = y3 + 2.0 * a;
  o_a = sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3;
  p_a = y3 + 2.0 * a;
  q_a = y3 + 2.0 * a;
  r_a = y3 + 2.0 * a;
  s_a = y3 + 2.0 * a;
  s_a = (sqrt((b_y1 * b_y1 + y2 * y2) + s_a * s_a) + y3) + 2.0 * a;
  t_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  u_a = y3 + 2.0 * a;
  v_a = y3 + 2.0 * a;
  w_a = y3 + 2.0 * a;
  x_a = y3 + 2.0 * a;
  y_a = y3 + 2.0 * a;
  ab_a = y3 + 2.0 * a;
  bb_a = y3 + 2.0 * a;
  cb_a = y3 + 2.0 * a;
  db_a = y3 + 2.0 * a;
  db_a = (sqrt((b_y1 * b_y1 + y2 * y2) + db_a * db_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  eb_a = y3 + 2.0 * a;
  fb_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  gb_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  hb_a = y3 + 2.0 * a;
  ib_a = y3 + 2.0 * a;
  jb_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  kb_a = y3 + 2.0 * a;
  c_x = sin(b);
  lb_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  d_x = 1.0 / tan(b);
  mb_a = y3 + 2.0 * a;
  mb_a = (sqrt((b_y1 * b_y1 + y2 * y2) + mb_a * mb_a) + y3) + 2.0 * a;
  nb_a = y3 + 2.0 * a;
  ob_a = y3 + 2.0 * a;
  pb_a = y3 + 2.0 * a;
  qb_a = y3 + 2.0 * a;
  rb_a = y3 + 2.0 * a;
  sb_a = y3 + 2.0 * a;
  tb_a = y3 + 2.0 * a;
  ub_a = y3 + 2.0 * a;
  vb_a = y3 + 2.0 * a;
  vb_a = (sqrt((b_y1 * b_y1 + y2 * y2) + vb_a * vb_a) + y3) + 2.0 * a;
  wb_a = y3 + 2.0 * a;
  xb_a = y3 + 2.0 * a;
  yb_a = y3 + 2.0 * a;
  ac_a = y3 + 2.0 * a;
  bc_a = y3 + 2.0 * a;
  bc_a = (sqrt((b_y1 * b_y1 + y2 * y2) + bc_a * bc_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  cc_a = y3 + 2.0 * a;
  dc_a = y3 + 2.0 * a;
  ec_a = y3 + 2.0 * a;
  fc_a = y3 + 2.0 * a;
  gc_a = y3 + 2.0 * a;
  hc_a = y3 + 2.0 * a;
  ic_a = y3 + 2.0 * a;
  jc_a = y3 + 2.0 * a;
  kc_a = y3 + 2.0 * a;
  lc_a = y3 + 2.0 * a;
  mc_a = y3 + 2.0 * a;
  nc_a = y3 + 2.0 * a;
  nc_a = (sqrt((b_y1 * b_y1 + y2 * y2) + nc_a * nc_a) + y3) + 2.0 * a;
  oc_a = y3 + 2.0 * a;
  pc_a = y3 + 2.0 * a;
  qc_a = y3 + 2.0 * a;
  rc_a = y3 + 2.0 * a;
  sc_a = y3 + 2.0 * a;
  tc_a = y3 + 2.0 * a;
  uc_a = y3 + 2.0 * a;
  vc_a = y3 + 2.0 * a;
  vc_a = (sqrt((b_y1 * b_y1 + y2 * y2) + vc_a * vc_a) + y3) + 2.0 * a;
  wc_a = y3 + 2.0 * a;
  xc_a = y3 + 2.0 * a;
  yc_a = y3 + 2.0 * a;
  ad_a = y3 + 2.0 * a;
  bd_a = y3 + 2.0 * a;
  cd_a = y3 + 2.0 * a;
  cd_a = (b_y1 * b_y1 + y2 * y2) + cd_a * cd_a;
  dd_a = y3 + 2.0 * a;
  ed_a = y3 + 2.0 * a;
  fd_a = y3 + 2.0 * a;
  gd_a = y3 + 2.0 * a;
  hd_a = y3 + 2.0 * a;
  id_a = y3 + 2.0 * a;
  jd_a = y3 + 2.0 * a;
  kd_a = y3 + 2.0 * a;
  ld_a = y3 + 2.0 * a;
  ld_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ld_a * ld_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  md_a = y3 + 2.0 * a;
  nd_a = y3 + 2.0 * a;
  od_a = y3 + 2.0 * a;
  pd_a = y3 + 2.0 * a;
  qd_a = y3 + 2.0 * a;
  rd_a = y3 + 2.0 * a;
  sd_a = y3 + 2.0 * a;
  td_a = y3 + 2.0 * a;
  ud_a = y3 + 2.0 * a;
  ud_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ud_a * ud_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  vd_a = y3 + 2.0 * a;
  wd_a = y3 + 2.0 * a;
  xd_a = y3 + 2.0 * a;
  yd_a = y3 + 2.0 * a;
  ae_a = y3 + 2.0 * a;
  be_a = y3 + 2.0 * a;
  ce_a = y3 + 2.0 * a;
  de_a = y3 + 2.0 * a;
  ee_a = y3 + 2.0 * a;
  fe_a = y3 + 2.0 * a;
  ge_a = y3 + 2.0 * a;
  ge_a = (b_y1 * b_y1 + y2 * y2) + ge_a * ge_a;
  he_a = y3 + 2.0 * a;
  ie_a = y3 + 2.0 * a;
  je_a = y3 + 2.0 * a;
  ke_a = y3 + 2.0 * a;
  le_a = y3 + 2.0 * a;
  me_a = y3 + 2.0 * a;
  ne_a = sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3;
  oe_a = y3 + 2.0 * a;
  pe_a = y3 + 2.0 * a;
  qe_a = y3 + 2.0 * a;
  re_a = y3 + 2.0 * a;
  re_a = (sqrt((b_y1 * b_y1 + y2 * y2) + re_a * re_a) + y3) + 2.0 * a;
  se_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  te_a = y3 + 2.0 * a;
  ue_a = y3 + 2.0 * a;
  ve_a = y3 + 2.0 * a;
  we_a = y3 + 2.0 * a;
  xe_a = y3 + 2.0 * a;
  ye_a = y3 + 2.0 * a;
  af_a = y3 + 2.0 * a;
  bf_a = y3 + 2.0 * a;
  cf_a = y3 + 2.0 * a;
  df_a = y3 + 2.0 * a;
  ef_a = y3 + 2.0 * a;
  ff_a = y3 + 2.0 * a;
  ff_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ff_a * ff_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  gf_a = y3 + 2.0 * a;
  e_x = 1.0 / tan(b);
  hf_a = y3 + 2.0 * a;
  if_a = y3 + 2.0 * a;
  f_x = 1.0 / tan(b);
  jf_a = y3 + 2.0 * a;
  kf_a = y3 + 2.0 * a;
  lf_a = y3 + 2.0 * a;
  lf_a = (sqrt((b_y1 * b_y1 + y2 * y2) + lf_a * lf_a) + y3) + 2.0 * a;
  mf_a = y3 + 2.0 * a;
  nf_a = y3 + 2.0 * a;
  of_a = y3 + 2.0 * a;
  pf_a = y3 + 2.0 * a;
  qf_a = y3 + 2.0 * a;
  rf_a = y3 + 2.0 * a;
  sf_a = y3 + 2.0 * a;
  tf_a = y3 + 2.0 * a;
  uf_a = y3 + 2.0 * a;
  vf_a = y3 + 2.0 * a;
  vf_a = (sqrt((b_y1 * b_y1 + y2 * y2) + vf_a * vf_a) + y3) + 2.0 * a;
  wf_a = y3 + 2.0 * a;
  xf_a = y3 + 2.0 * a;
  yf_a = y3 + 2.0 * a;
  ag_a = y3 + 2.0 * a;
  bg_a = y3 + 2.0 * a;
  bg_a = (sqrt((b_y1 * b_y1 + y2 * y2) + bg_a * bg_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  cg_a = y3 + 2.0 * a;
  dg_a = y3 + 2.0 * a;
  eg_a = y3 + 2.0 * a;
  fg_a = y3 + 2.0 * a;
  g_x = cos(b);
  gg_a = y3 + 2.0 * a;
  hg_a = y3 + 2.0 * a;
  ig_a = y3 + 2.0 * a;
  jg_a = y3 + 2.0 * a;
  kg_a = y3 + 2.0 * a;
  lg_a = y3 + 2.0 * a;
  mg_a = y3 + 2.0 * a;
  mg_a = (sqrt((b_y1 * b_y1 + y2 * y2) + mg_a * mg_a) + y3) + 2.0 * a;
  ng_a = y3 + 2.0 * a;
  og_a = y3 + 2.0 * a;
  pg_a = y3 + 2.0 * a;
  qg_a = y3 + 2.0 * a;
  rg_a = y3 + 2.0 * a;
  sg_a = y3 + 2.0 * a;
  tg_a = y3 + 2.0 * a;
  ug_a = y3 + 2.0 * a;
  vg_a = y3 + 2.0 * a;
  wg_a = y3 + 2.0 * a;
  xg_a = y3 + 2.0 * a;
  yg_a = y3 + 2.0 * a;
  ah_a = y3 + 2.0 * a;
  bh_a = y3 + 2.0 * a;
  ch_a = y3 + 2.0 * a;
  dh_a = y3 + 2.0 * a;
  eh_a = y3 + 2.0 * a;
  eh_a = (sqrt((b_y1 * b_y1 + y2 * y2) + eh_a * eh_a) + y3) + 2.0 * a;
  fh_a = y3 + 2.0 * a;
  gh_a = y3 + 2.0 * a;
  gh_a = (b_y1 * b_y1 + y2 * y2) + gh_a * gh_a;
  hh_a = y3 + 2.0 * a;
  ih_a = y3 + 2.0 * a;
  jh_a = y3 + 2.0 * a;
  kh_a = y3 + 2.0 * a;
  kh_a = (sqrt((b_y1 * b_y1 + y2 * y2) + kh_a * kh_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  lh_a = y3 + 2.0 * a;
  mh_a = y3 + 2.0 * a;
  nh_a = y3 + 2.0 * a;
  oh_a = y3 + 2.0 * a;
  ph_a = y3 + 2.0 * a;
  qh_a = y3 + 2.0 * a;
  rh_a = y3 + 2.0 * a;
  sh_a = y3 + 2.0 * a;
  th_a = y3 + 2.0 * a;
  uh_a = y3 + 2.0 * a;
  vh_a = y3 + 2.0 * a;
  wh_a = y3 + 2.0 * a;
  xh_a = y3 + 2.0 * a;
  yh_a = y3 + 2.0 * a;
  ai_a = y3 + 2.0 * a;
  bi_a = y3 + 2.0 * a;
  ci_a = y3 + 2.0 * a;
  di_a = y3 + 2.0 * a;
  ei_a = y3 + 2.0 * a;
  fi_a = y3 + 2.0 * a;
  gi_a = y3 + 2.0 * a;
  hi_a = y3 + 2.0 * a;
  ii_a = y3 + 2.0 * a;
  ji_a = y3 + 2.0 * a;
  ki_a = y3 + 2.0 * a;
  li_a = y3 + 2.0 * a;
  mi_a = y3 + 2.0 * a;
  mi_a = (sqrt((b_y1 * b_y1 + y2 * y2) + mi_a * mi_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  ni_a = y3 + 2.0 * a;
  oi_a = y3 + 2.0 * a;
  pi_a = y3 + 2.0 * a;
  qi_a = y3 + 2.0 * a;
  ri_a = y3 + 2.0 * a;
  si_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  ti_a = y3 + 2.0 * a;
  ui_a = y3 + 2.0 * a;
  vi_a = y3 + 2.0 * a;
  wi_a = y3 + 2.0 * a;
  xi_a = y3 + 2.0 * a;
  yi_a = y3 + 2.0 * a;
  aj_a = y3 + 2.0 * a;
  bj_a = y3 + 2.0 * a;
  cj_a = y3 + 2.0 * a;
  cj_a = (sqrt((b_y1 * b_y1 + y2 * y2) + cj_a * cj_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  dj_a = y3 + 2.0 * a;
  ej_a = y3 + 2.0 * a;
  ej_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ej_a * ej_a) + y3) + 2.0 * a;
  fj_a = y3 + 2.0 * a;
  gj_a = y3 + 2.0 * a;
  hj_a = y3 + 2.0 * a;
  ij_a = y3 + 2.0 * a;
  jj_a = y3 + 2.0 * a;
  jj_a = (sqrt((b_y1 * b_y1 + y2 * y2) + jj_a * jj_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  kj_a = y3 + 2.0 * a;
  lj_a = y3 + 2.0 * a;
  mj_a = y3 + 2.0 * a;
  nj_a = y3 + 2.0 * a;
  oj_a = y3 + 2.0 * a;
  pj_a = y3 + 2.0 * a;
  qj_a = y3 + 2.0 * a;
  rj_a = y3 + 2.0 * a;
  sj_a = y3 + 2.0 * a;
  sj_a = (b_y1 * b_y1 + y2 * y2) + sj_a * sj_a;
  tj_a = y3 + 2.0 * a;
  tj_a = (sqrt((b_y1 * b_y1 + y2 * y2) + tj_a * tj_a) + y3) + 2.0 * a;
  uj_a = y3 + 2.0 * a;
  vj_a = y3 + 2.0 * a;
  wj_a = y3 + 2.0 * a;
  xj_a = y3 + 2.0 * a;
  yj_a = y3 + 2.0 * a;
  ak_a = y3 + 2.0 * a;
  bk_a = y3 + 2.0 * a;
  ck_a = y3 + 2.0 * a;
  dk_a = y3 + 2.0 * a;
  dk_a = (sqrt((b_y1 * b_y1 + y2 * y2) + dk_a * dk_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  ek_a = y3 + 2.0 * a;
  fk_a = y3 + 2.0 * a;
  gk_a = y3 + 2.0 * a;
  hk_a = y3 + 2.0 * a;
  ik_a = y3 + 2.0 * a;
  jk_a = y3 + 2.0 * a;
  kk_a = y3 + 2.0 * a;
  lk_a = y3 + 2.0 * a;
  mk_a = y3 + 2.0 * a;
  nk_a = y3 + 2.0 * a;
  ok_a = y3 + 2.0 * a;
  pk_a = y3 + 2.0 * a;
  pk_a = (sqrt((b_y1 * b_y1 + y2 * y2) + pk_a * pk_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  qk_a = y3 + 2.0 * a;
  rk_a = y3 + 2.0 * a;
  sk_a = y3 + 2.0 * a;
  tk_a = y3 + 2.0 * a;
  uk_a = y3 + 2.0 * a;
  vk_a = y3 + 2.0 * a;
  vk_a = (b_y1 * b_y1 + y2 * y2) + vk_a * vk_a;
  *e11 = (B1 * (0.125 * ((((2.0 - 2.0 * nu) * ((((2.0 * y2 / (b_y1 * b_y1) /
    (1.0 + y2 * y2 / (b_y1 * b_y1)) - y2 / (b_a * b_a) * cos(b) / (1.0 + y2 * y2
    / (c_a * c_a))) + (y2 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) /
                       (b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b))
                       * b_y1 - y2 * sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) *
                       sin(b) / (d_a * d_a) * (2.0 * b_y1 * cos(b) - y3 * sin(b)))
    / (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * (x * x) / (e_a *
    e_a))) - y2 / (f_a * f_a) * cos(b) / (1.0 + y2 * y2 / (g_a * g_a))) + (y2 /
    sqrt((b_y1 * b_y1 + y2 * y2) + h_a * h_a) * sin(b) / (b_y1 * (b_y1 * cos(b)
    + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b)) * b_y1 - y2 * sqrt((b_y1 *
    b_y1 + y2 * y2) + i_a * i_a) * sin(b) / (j_a * j_a) * (2.0 * b_y1 * cos(b) +
    (y3 + 2.0 * a) * sin(b))) / (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2) + k_a *
    k_a) * (b_x * b_x) / (l_a * l_a))) - y2 * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2)
    + y3 * y3) / (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3) + 1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + m_a * m_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + n_a
    * n_a) + y3) + 2.0 * a))) - b_y1 * y2 * (((-1.0 / powd_snf((b_y1 * b_y1 +
    y2 * y2) + y3 * y3, 1.5) / (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3) *
    b_y1 - 1.0 / ((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (o_a * o_a) * b_y1) - 1.0
    / powd_snf((b_y1 * b_y1 + y2 * y2) + p_a * p_a, 1.5) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + q_a * q_a) + y3) + 2.0 * a) * b_y1) - 1.0 / ((b_y1 * b_y1
    + y2 * y2) + r_a * r_a) / (s_a * s_a) * b_y1)) - y2 * cos(b) * ((((((1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) * b_y1 - 1.0) / sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) -
    b_y1 * sin(b)) - y3 * cos(b)) - (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) *
    sin(b) - b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) *
    b_y1) - (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) - b_y1) / sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (t_a * t_a) * (1.0 / sqrt((b_y1 * b_y1
    + y2 * y2) + y3 * y3) * b_y1 - sin(b))) + (1.0 / sqrt((b_y1 * b_y1 + y2 * y2)
    + u_a * u_a) * sin(b) * b_y1 - 1.0) / sqrt((b_y1 * b_y1 + y2 * y2) + v_a *
    v_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + w_a * w_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b))) - (sqrt((b_y1 * b_y1 + y2 * y2) + x_a * x_a) * sin(b) -
    b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + y_a * y_a, 1.5) / ((sqrt((b_y1
    * b_y1 + y2 * y2) + ab_a * ab_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))
              * b_y1) - (sqrt((b_y1 * b_y1 + y2 * y2) + bb_a * bb_a) * sin(b) -
              b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + cb_a * cb_a) / (db_a * db_a)
             * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + eb_a * eb_a) * b_y1 - sin(b))))
                / M_PI / (1.0 - nu) + 0.25 * ((((((((((((-2.0 +
    2.0 * nu) * (1.0 - 2.0 * nu) * ((y2 / (b_y1 * b_y1) / (1.0 + y2 * y2 / (b_y1
    * b_y1)) - y2 / (fb_a * fb_a) * cos(b) / (1.0 + y2 * y2 / (gb_a * gb_a))) +
    (y2 / sqrt((b_y1 * b_y1 + y2 * y2) + hb_a * hb_a) * sin(b) / (b_y1 * (b_y1 *
    cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b)) * b_y1 - y2 * sqrt
     ((b_y1 * b_y1 + y2 * y2) + ib_a * ib_a) * sin(b) / (jb_a * jb_a) * (2.0 *
    b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b))) / (1.0 + y2 * y2 * ((b_y1 * b_y1 +
    y2 * y2) + kb_a * kb_a) * (c_x * c_x) / (lb_a * lb_a))) * (d_x * d_x) - (1.0
    - 2.0 * nu) * y2 / (mb_a * mb_a) * (((1.0 - 2.0 * nu) - a / sqrt((b_y1 *
    b_y1 + y2 * y2) + nb_a * nb_a)) * (1.0 / tan(b)) - b_y1 / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + ob_a * ob_a) + y3) + 2.0 * a) * (nu + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + pb_a * pb_a))) / sqrt((b_y1 * b_y1 + y2 * y2) + qb_a *
    qb_a) * b_y1) + (1.0 - 2.0 * nu) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    rb_a * rb_a) + y3) + 2.0 * a) * (((a / powd_snf((b_y1 * b_y1 + y2 * y2) +
    sb_a * sb_a, 1.5) * b_y1 * (1.0 / tan(b)) - 1.0 / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + tb_a * tb_a) + y3) + 2.0 * a) * (nu + a / sqrt((b_y1 * b_y1 + y2 * y2)
    + ub_a * ub_a))) + b_y1 * b_y1 / (vb_a * vb_a) * (nu + a / sqrt((b_y1 * b_y1
    + y2 * y2) + wb_a * wb_a)) / sqrt((b_y1 * b_y1 + y2 * y2) + xb_a * xb_a)) +
    b_y1 * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + yb_a * yb_a) + y3) + 2.0 * a)
    * a / powd_snf((b_y1 * b_y1 + y2 * y2) + ac_a * ac_a, 1.5))) - (1.0 - 2.0
    * nu) * y2 * cos(b) * (1.0 / tan(b)) / (bc_a * bc_a) * (cos(b) + a / sqrt
    ((b_y1 * b_y1 + y2 * y2) + cc_a * cc_a)) * (1.0 / sqrt((b_y1 * b_y1 + y2 *
    y2) + dc_a * dc_a) * b_y1 - sin(b))) - (1.0 - 2.0 * nu) * y2 * cos(b) * (1.0
    / tan(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ec_a * ec_a) - b_y1 * sin(b)) +
                 (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 *
    y2) + fc_a * fc_a, 1.5) * b_y1) - 3.0 * a * y2 * (y3 + a) * (1.0 / tan(b)) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + gc_a * gc_a, 2.5) * b_y1) - y2 * (y3 +
    a) / powd_snf((b_y1 * b_y1 + y2 * y2) + hc_a * hc_a, 1.5) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + ic_a * ic_a) + y3) + 2.0 * a) * (((-1.0 + 2.0 * nu) * (1.0
    / tan(b)) + b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + jc_a * jc_a) + y3) + 2.0
                        * a) * (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 * y2) +
    kc_a * kc_a))) + a * b_y1 / ((b_y1 * b_y1 + y2 * y2) + lc_a * lc_a)) * b_y1)
    - y2 * (y3 + a) / ((b_y1 * b_y1 + y2 * y2) + mc_a * mc_a) / (nc_a * nc_a) *
    (((-1.0 + 2.0 * nu) * (1.0 / tan(b)) + b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + oc_a * oc_a) + y3) + 2.0 * a) * (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 *
    y2) + pc_a * pc_a))) + a * b_y1 / ((b_y1 * b_y1 + y2 * y2) + qc_a * qc_a)) *
    b_y1) + y2 * (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) + rc_a * rc_a) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + sc_a * sc_a) + y3) + 2.0 * a) * ((((1.0 / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + tc_a * tc_a) + y3) + 2.0 * a) * (2.0 * nu + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + uc_a * uc_a)) - b_y1 * b_y1 / (vc_a * vc_a) *
    (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + wc_a * wc_a)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + xc_a * xc_a)) - b_y1 * b_y1 / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + yc_a * yc_a) + y3) + 2.0 * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2)
    + ad_a * ad_a, 1.5)) + a / ((b_y1 * b_y1 + y2 * y2) + bd_a * bd_a)) - 2.0 *
    a * (b_y1 * b_y1) / (cd_a * cd_a))) - y2 * (y3 + a) / powd_snf((b_y1 *
    b_y1 + y2 * y2) + dd_a * dd_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ed_a *
    ed_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + fd_a * fd_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    (((sqrt((b_y1 * b_y1 + y2 * y2) + gd_a * gd_a) * cos(b) + y3) + 2.0 * a) *
     ((1.0 - 2.0 * nu) * cos(b) - a / sqrt((b_y1 * b_y1 + y2 * y2) + hd_a * hd_a))
     * (1.0 / tan(b)) + (2.0 - 2.0 * nu) * (sqrt((b_y1 * b_y1 + y2 * y2) + id_a *
    id_a) * sin(b) - b_y1) * cos(b)) - a * (y3 + 2.0 * a) * cos(b) * (1.0 / tan
    (b)) / ((b_y1 * b_y1 + y2 * y2) + jd_a * jd_a)) * b_y1) - y2 * (y3 + a) /
             sqrt((b_y1 * b_y1 + y2 * y2) + kd_a * kd_a) / (ld_a * ld_a) * (cos
              (b) / ((sqrt((b_y1 * b_y1 + y2 * y2) + md_a * md_a) - b_y1 * sin(b))
                     + (y3 + 2.0 * a) * cos(b)) * (((sqrt((b_y1 * b_y1 + y2 * y2)
    + nd_a * nd_a) * cos(b) + y3) + 2.0 * a) * ((1.0 - 2.0 * nu) * cos(b) - a /
    sqrt((b_y1 * b_y1 + y2 * y2) + od_a * od_a)) * (1.0 / tan(b)) + (2.0 - 2.0 *
    nu) * (sqrt((b_y1 * b_y1 + y2 * y2) + pd_a * pd_a) * sin(b) - b_y1) * cos(b))
              - a * (y3 + 2.0 * a) * cos(b) * (1.0 / tan(b)) / ((b_y1 * b_y1 +
    y2 * y2) + qd_a * qd_a)) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + rd_a * rd_a)
              * b_y1 - sin(b))) + y2 * (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) +
             sd_a * sd_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + td_a * td_a) - b_y1
              * sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((-cos(b) / (ud_a * ud_a) *
              (((sqrt((b_y1 * b_y1 + y2 * y2) + vd_a * vd_a) * cos(b) + y3) +
                2.0 * a) * ((1.0 - 2.0 * nu) * cos(b) - a / sqrt((b_y1 * b_y1 +
    y2 * y2) + wd_a * wd_a)) * (1.0 / tan(b)) + (2.0 - 2.0 * nu) * (sqrt((b_y1 *
    b_y1 + y2 * y2) + xd_a * xd_a) * sin(b) - b_y1) * cos(b)) * (1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + yd_a * yd_a) * b_y1 - sin(b)) + cos(b) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + ae_a * ae_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b)) * ((1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + be_a * be_a) * cos(b) *
                b_y1 * ((1.0 - 2.0 * nu) * cos(b) - a / sqrt((b_y1 * b_y1 + y2 *
    y2) + ce_a * ce_a)) * (1.0 / tan(b)) + ((sqrt((b_y1 * b_y1 + y2 * y2) + de_a
    * de_a) * cos(b) + y3) + 2.0 * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2)
    + ee_a * ee_a, 1.5) * b_y1 * (1.0 / tan(b))) + (2.0 - 2.0 * nu) * (1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + fe_a * fe_a) * sin(b) * b_y1 - 1.0) * cos(b)))
             + 2.0 * a * (y3 + 2.0 * a) * cos(b) * (1.0 / tan(b)) / (ge_a * ge_a)
             * b_y1)) / M_PI / (1.0 - nu)) + B2 * (0.125 *
           (((((((((((-1.0 + 2.0 * nu) * ((1.0 / sqrt((b_y1 * b_y1 + y2 * y2) +
    y3 * y3) * b_y1 / (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3) + 1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + he_a * he_a) * b_y1 / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + ie_a * ie_a) + y3) + 2.0 * a)) - cos(b) * ((1.0 / sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) * b_y1 - sin(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) + (1.0 / sqrt((b_y1 * b_y1 + y2 *
    y2) + je_a * je_a) * b_y1 - sin(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ke_a *
    ke_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)))) + 2.0 * b_y1 * (1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (sqrt((b_y1 * b_y1 + y2 * y2) + y3
    * y3) - y3) + 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + le_a * le_a) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + me_a * me_a) + y3) + 2.0 * a))) + b_y1 * b_y1 *
                    (((-1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5)
                       / (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3) * b_y1 -
                       1.0 / ((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (ne_a * ne_a)
                       * b_y1) - 1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) +
    oe_a * oe_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + pe_a * pe_a) + y3) +
    2.0 * a) * b_y1) - 1.0 / ((b_y1 * b_y1 + y2 * y2) + qe_a * qe_a) / (re_a *
    re_a) * b_y1)) + cos(b) * (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b)
    - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / ((sqrt((b_y1 * b_y1 + y2
    * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b))) + (b_y1 * cos(b) - y3 *
    sin(b)) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) * b_y1 -
               1.0) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b))) - (b_y1 * cos(b)
    - y3 * sin(b)) * (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) - b_y1) /
                 powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) * b_y1)
                - (b_y1 * cos(b) - y3 * sin(b)) * (sqrt((b_y1 * b_y1 + y2 * y2)
    + y3 * y3) * sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) /
                (se_a * se_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) *
    b_y1 - sin(b))) + cos(b) * (sqrt((b_y1 * b_y1 + y2 * y2) + te_a * te_a) *
    sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + ue_a * ue_a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + ve_a * ve_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)))
              + (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / sqrt((b_y1 *
    b_y1 + y2 * y2) + we_a * we_a) * sin(b) * b_y1 - 1.0) / sqrt((b_y1 * b_y1 +
    y2 * y2) + xe_a * xe_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ye_a * ye_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) - (b_y1 * cos(b) + (y3 + 2.0 * a)
              * sin(b)) * (sqrt((b_y1 * b_y1 + y2 * y2) + af_a * af_a) * sin(b)
              - b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + bf_a * bf_a, 1.5) /
             ((sqrt((b_y1 * b_y1 + y2 * y2) + cf_a * cf_a) - b_y1 * sin(b)) +
              (y3 + 2.0 * a) * cos(b)) * b_y1) - (b_y1 * cos(b) + (y3 + 2.0 * a)
             * sin(b)) * (sqrt((b_y1 * b_y1 + y2 * y2) + df_a * df_a) * sin(b) -
             b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + ef_a * ef_a) / (ff_a * ff_a)
            * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + gf_a * gf_a) * b_y1 - sin(b)))
           / M_PI / (1.0 - nu) + 0.25 * (((((((((((1.0 - 2.0 * nu)
    * (((2.0 - 2.0 * nu) * (e_x * e_x) + nu) / sqrt((b_y1 * b_y1 + y2 * y2) +
    hf_a * hf_a) * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + if_a * if_a) + y3) +
    2.0 * a) - ((2.0 - 2.0 * nu) * (f_x * f_x) + 1.0) * cos(b) * (1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + jf_a * jf_a) * b_y1 - sin(b)) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + kf_a * kf_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)))
    - (1.0 - 2.0 * nu) / (lf_a * lf_a) * (((((-1.0 + 2.0 * nu) * b_y1 * (1.0 /
    tan(b)) + nu * (y3 + 2.0 * a)) - a) + a * b_y1 * (1.0 / tan(b)) / sqrt((b_y1
    * b_y1 + y2 * y2) + mf_a * mf_a)) + b_y1 * b_y1 / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + nf_a * nf_a) + y3) + 2.0 * a) * (nu + a / sqrt((b_y1 * b_y1 + y2 * y2)
    + of_a * of_a))) / sqrt((b_y1 * b_y1 + y2 * y2) + pf_a * pf_a) * b_y1) +
    (1.0 - 2.0 * nu) / ((sqrt((b_y1 * b_y1 + y2 * y2) + qf_a * qf_a) + y3) + 2.0
                        * a) * ((((((-1.0 + 2.0 * nu) * (1.0 / tan(b)) + a *
    (1.0 / tan(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + rf_a * rf_a)) - a * (b_y1 *
    b_y1) * (1.0 / tan(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + sf_a * sf_a,
    1.5)) + 2.0 * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + tf_a * tf_a) + y3) +
    2.0 * a) * (nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + uf_a * uf_a))) -
    powd_snf(b_y1, 3.0) / (vf_a * vf_a) * (nu + a / sqrt((b_y1 * b_y1 + y2 *
    y2) + wf_a * wf_a)) / sqrt((b_y1 * b_y1 + y2 * y2) + xf_a * xf_a)) -
    powd_snf(b_y1, 3.0) / ((sqrt((b_y1 * b_y1 + y2 * y2) + yf_a * yf_a) + y3)
    + 2.0 * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + ag_a * ag_a, 1.5))) +
    (1.0 - 2.0 * nu) * (1.0 / tan(b)) / (bg_a * bg_a) * ((b_y1 * cos(b) + (y3 +
    2.0 * a) * sin(b)) * cos(b) - a * (sqrt((b_y1 * b_y1 + y2 * y2) + cg_a *
    cg_a) * sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + dg_a * dg_a) / cos(b))
    * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + eg_a * eg_a) * b_y1 - sin(b))) -
    (1.0 - 2.0 * nu) * (1.0 / tan(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + fg_a *
    fg_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((g_x * g_x - a * (1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + gg_a * gg_a) * sin(b) * b_y1 - 1.0) / sqrt
    ((b_y1 * b_y1 + y2 * y2) + hg_a * hg_a) / cos(b)) + a * (sqrt((b_y1 * b_y1 +
    y2 * y2) + ig_a * ig_a) * sin(b) - b_y1) / powd_snf((b_y1 * b_y1 + y2 *
    y2) + jg_a * jg_a, 1.5) / cos(b) * b_y1)) - a * (y3 + a) * (1.0 / tan(b)) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + kg_a * kg_a, 1.5)) + 3.0 * a * (b_y1 *
    b_y1) * (y3 + a) * (1.0 / tan(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) +
    lg_a * lg_a, 2.5)) - (y3 + a) / (mg_a * mg_a) * (((2.0 * nu + 1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + ng_a * ng_a) * ((1.0 - 2.0 * nu) * b_y1 * (1.0 /
    tan(b)) + a)) - b_y1 * b_y1 / sqrt((b_y1 * b_y1 + y2 * y2) + og_a * og_a) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + pg_a * pg_a) + y3) + 2.0 * a) * (2.0 * nu +
    a / sqrt((b_y1 * b_y1 + y2 * y2) + qg_a * qg_a))) - a * (b_y1 * b_y1) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + rg_a * rg_a, 1.5)) / sqrt((b_y1 * b_y1
    + y2 * y2) + sg_a * sg_a) * b_y1) + (y3 + a) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + tg_a * tg_a) + y3) + 2.0 * a) * (((((((-1.0 / powd_snf((b_y1 * b_y1 +
    y2 * y2) + ug_a * ug_a, 1.5) * ((1.0 - 2.0 * nu) * b_y1 * (1.0 / tan(b)) + a)
    * b_y1 + 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + vg_a * vg_a) * (1.0 - 2.0 * nu)
    * (1.0 / tan(b))) - 2.0 * b_y1 / sqrt((b_y1 * b_y1 + y2 * y2) + wg_a * wg_a)
    / ((sqrt((b_y1 * b_y1 + y2 * y2) + xg_a * xg_a) + y3) + 2.0 * a) * (2.0 * nu
    + a / sqrt((b_y1 * b_y1 + y2 * y2) + yg_a * yg_a))) + powd_snf(b_y1, 3.0)
    / powd_snf((b_y1 * b_y1 + y2 * y2) + ah_a * ah_a, 1.5) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + bh_a * bh_a) + y3) + 2.0 * a) * (2.0 * nu + a / sqrt((b_y1
    * b_y1 + y2 * y2) + ch_a * ch_a))) + powd_snf(b_y1, 3.0) / ((b_y1 * b_y1
    + y2 * y2) + dh_a * dh_a) / (eh_a * eh_a) * (2.0 * nu + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + fh_a * fh_a))) + powd_snf(b_y1, 3.0) / (gh_a * gh_a) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + hh_a * hh_a) + y3) + 2.0 * a) * a) - 2.0 *
    a / powd_snf((b_y1 * b_y1 + y2 * y2) + ih_a * ih_a, 1.5) * b_y1) + 3.0 *
    a * powd_snf(b_y1, 3.0) / powd_snf((b_y1 * b_y1 + y2 * y2) + jh_a *
    jh_a, 2.5))) - (y3 + a) * (1.0 / tan(b)) / (kh_a * kh_a) * ((-cos(b) * sin(b)
    + a * b_y1 * (y3 + 2.0 * a) / powd_snf((b_y1 * b_y1 + y2 * y2) + lh_a *
    lh_a, 1.5) / cos(b)) + (sqrt((b_y1 * b_y1 + y2 * y2) + mh_a * mh_a) * sin(b)
    - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + nh_a * nh_a) * ((2.0 - 2.0 * nu) *
    cos(b) - ((sqrt((b_y1 * b_y1 + y2 * y2) + oh_a * oh_a) * cos(b) + y3) + 2.0 *
              a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ph_a * ph_a) - b_y1 * sin(b))
                    + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt((b_y1 * b_y1 +
    y2 * y2) + qh_a * qh_a) / cos(b)))) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) +
    rh_a * rh_a) * b_y1 - sin(b))) + (y3 + a) * (1.0 / tan(b)) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + sh_a * sh_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
            ((((a * (y3 + 2.0 * a) / powd_snf((b_y1 * b_y1 + y2 * y2) + th_a *
    th_a, 1.5) / cos(b) - 3.0 * a * (b_y1 * b_y1) * (y3 + 2.0 * a) / powd_snf
                ((b_y1 * b_y1 + y2 * y2) + uh_a * uh_a, 2.5) / cos(b)) + (1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + vh_a * vh_a) * sin(b) * b_y1 - 1.0) / sqrt
               ((b_y1 * b_y1 + y2 * y2) + wh_a * wh_a) * ((2.0 - 2.0 * nu) * cos
    (b) - ((sqrt((b_y1 * b_y1 + y2 * y2) + xh_a * xh_a) * cos(b) + y3) + 2.0 * a)
    / ((sqrt((b_y1 * b_y1 + y2 * y2) + yh_a * yh_a) - b_y1 * sin(b)) + (y3 + 2.0
    * a) * cos(b)) * (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2) + ai_a * ai_a) /
                      cos(b)))) - (sqrt((b_y1 * b_y1 + y2 * y2) + bi_a * bi_a) *
    sin(b) - b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + ci_a * ci_a, 1.5) *
              ((2.0 - 2.0 * nu) * cos(b) - ((sqrt((b_y1 * b_y1 + y2 * y2) + di_a
    * di_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ei_a *
    ei_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + fi_a * fi_a) / cos(b))) * b_y1) + (sqrt((b_y1 * b_y1 + y2 *
    y2) + gi_a * gi_a) * sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + hi_a *
              hi_a) * ((-1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + ii_a * ii_a) * cos
                        (b) * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + ji_a *
    ji_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + ki_a * ki_a) / cos(b)) + ((sqrt((b_y1 * b_y1 + y2 * y2) +
    li_a * li_a) * cos(b) + y3) + 2.0 * a) / (mi_a * mi_a) * (1.0 + a / sqrt
    ((b_y1 * b_y1 + y2 * y2) + ni_a * ni_a) / cos(b)) * (1.0 / sqrt((b_y1 * b_y1
    + y2 * y2) + oi_a * oi_a) * b_y1 - sin(b))) + ((sqrt((b_y1 * b_y1 + y2 * y2)
    + pi_a * pi_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    qi_a * qi_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf
                       ((b_y1 * b_y1 + y2 * y2) + ri_a * ri_a, 1.5) / cos(b) *
                       b_y1))) / M_PI / (1.0 - nu))) + B3 * (0.125
    * y2 * sin(b) * ((((((1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b)
    * b_y1 - 1.0) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) - (sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) * sin(b) - b_y1) / powd_snf((b_y1 * b_y1 + y2 *
    y2) + y3 * y3, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin
    (b)) - y3 * cos(b)) * b_y1) - (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin
    (b) - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (si_a * si_a) * (1.0
    / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * b_y1 - sin(b))) + (1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + ti_a * ti_a) * sin(b) * b_y1 - 1.0) / sqrt((b_y1 *
    b_y1 + y2 * y2) + ui_a * ui_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + vi_a *
    vi_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) - (sqrt((b_y1 * b_y1 + y2
    * y2) + wi_a * wi_a) * sin(b) - b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2)
    + xi_a * xi_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + yi_a * yi_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * b_y1) - (sqrt((b_y1 * b_y1 + y2 * y2) +
    aj_a * aj_a) * sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + bj_a * bj_a) /
                     (cj_a * cj_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + dj_a *
    dj_a) * b_y1 - sin(b))) / M_PI / (1.0 - nu) + 0.25 * ((((((1.0
    - 2.0 * nu) * (((-y2 / (ej_a * ej_a) * (1.0 + a / sqrt((b_y1 * b_y1 + y2 *
    y2) + fj_a * fj_a)) / sqrt((b_y1 * b_y1 + y2 * y2) + gj_a * gj_a) * b_y1 -
                     y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + hj_a * hj_a) + y3) +
    2.0 * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + ij_a * ij_a, 1.5) *
                     b_y1) + y2 * cos(b) / (jj_a * jj_a) * (cos(b) + a / sqrt
    ((b_y1 * b_y1 + y2 * y2) + kj_a * kj_a)) * (1.0 / sqrt((b_y1 * b_y1 + y2 *
    y2) + lj_a * lj_a) * b_y1 - sin(b))) + y2 * cos(b) / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + mj_a * mj_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * a /
                   powd_snf((b_y1 * b_y1 + y2 * y2) + nj_a * nj_a, 1.5) *
                   b_y1) + y2 * (y3 + a) / powd_snf((b_y1 * b_y1 + y2 * y2) +
    oj_a * oj_a, 1.5) * (a / ((b_y1 * b_y1 + y2 * y2) + pj_a * pj_a) + 1.0 /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + qj_a * qj_a) + y3) + 2.0 * a)) * b_y1) - y2
    * (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) + rj_a * rj_a) * (-2.0 * a / (sj_a
    * sj_a) * b_y1 - 1.0 / (tj_a * tj_a) / sqrt((b_y1 * b_y1 + y2 * y2) + uj_a *
    uj_a) * b_y1)) - y2 * (y3 + a) * cos(b) / powd_snf((b_y1 * b_y1 + y2 * y2)
    + vj_a * vj_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + wj_a * wj_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((sqrt((b_y1 * b_y1 + y2 * y2) + xj_a *
    xj_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + yj_a *
    yj_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1
    * b_y1 + y2 * y2) + ak_a * ak_a)) + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 *
    y2) + bk_a * bk_a)) * b_y1) - y2 * (y3 + a) * cos(b) / sqrt((b_y1 * b_y1 +
    y2 * y2) + ck_a * ck_a) / (dk_a * dk_a) * (((sqrt((b_y1 * b_y1 + y2 * y2) +
    ek_a * ek_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    fk_a * fk_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + gk_a * gk_a)) + a * (y3 + 2.0 * a) / ((b_y1 *
    b_y1 + y2 * y2) + hk_a * hk_a)) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + ik_a
    * ik_a) * b_y1 - sin(b))) + y2 * (y3 + a) * cos(b) / sqrt((b_y1 * b_y1 + y2 *
    y2) + jk_a * jk_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + kk_a * kk_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((1.0 / sqrt((b_y1 * b_y1 + y2 * y2) +
    lk_a * lk_a) * cos(b) * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + mk_a * mk_a)
    - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + nk_a * nk_a)) - ((sqrt((b_y1 * b_y1 + y2 * y2) + ok_a *
    ok_a) * cos(b) + y3) + 2.0 * a) / (pk_a * pk_a) * (cos(b) + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + qk_a * qk_a)) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + rk_a
    * rk_a) * b_y1 - sin(b))) - ((sqrt((b_y1 * b_y1 + y2 * y2) + sk_a * sk_a) *
    cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + tk_a * tk_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 * b_y1 +
    y2 * y2) + uk_a * uk_a, 1.5) * b_y1) - 2.0 * a * (y3 + 2.0 * a) / (vk_a *
    vk_a) * b_y1)) / M_PI / (1.0 - nu));
  b_a = y3 + 2.0 * a;
  c_a = y3 + 2.0 * a;
  d_a = y3 + 2.0 * a;
  e_a = y3 + 2.0 * a;
  f_a = y3 + 2.0 * a;
  g_a = y3 + 2.0 * a;
  h_a = y3 + 2.0 * a;
  i_a = y3 + 2.0 * a;
  j_a = sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3;
  k_a = y3 + 2.0 * a;
  l_a = y3 + 2.0 * a;
  m_a = y3 + 2.0 * a;
  n_a = y3 + 2.0 * a;
  n_a = (sqrt((b_y1 * b_y1 + y2 * y2) + n_a * n_a) + y3) + 2.0 * a;
  o_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  p_a = y3 + 2.0 * a;
  q_a = y3 + 2.0 * a;
  r_a = y3 + 2.0 * a;
  s_a = y3 + 2.0 * a;
  s_a = (sqrt((b_y1 * b_y1 + y2 * y2) + s_a * s_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b);
  x = 1.0 / tan(b);
  t_a = y3 + 2.0 * a;
  u_a = y3 + 2.0 * a;
  b_x = 1.0 / tan(b);
  v_a = y3 + 2.0 * a;
  w_a = y3 + 2.0 * a;
  x_a = y3 + 2.0 * a;
  x_a = (sqrt((b_y1 * b_y1 + y2 * y2) + x_a * x_a) + y3) + 2.0 * a;
  y_a = y3 + 2.0 * a;
  ab_a = y3 + 2.0 * a;
  bb_a = y3 + 2.0 * a;
  cb_a = y3 + 2.0 * a;
  db_a = y3 + 2.0 * a;
  eb_a = y3 + 2.0 * a;
  fb_a = y3 + 2.0 * a;
  gb_a = y3 + 2.0 * a;
  hb_a = y3 + 2.0 * a;
  hb_a = (sqrt((b_y1 * b_y1 + y2 * y2) + hb_a * hb_a) + y3) + 2.0 * a;
  ib_a = y3 + 2.0 * a;
  jb_a = y3 + 2.0 * a;
  kb_a = y3 + 2.0 * a;
  lb_a = y3 + 2.0 * a;
  mb_a = y3 + 2.0 * a;
  mb_a = (sqrt((b_y1 * b_y1 + y2 * y2) + mb_a * mb_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  nb_a = y3 + 2.0 * a;
  ob_a = y3 + 2.0 * a;
  pb_a = y3 + 2.0 * a;
  qb_a = y3 + 2.0 * a;
  rb_a = y3 + 2.0 * a;
  sb_a = y3 + 2.0 * a;
  sb_a = (sqrt((b_y1 * b_y1 + y2 * y2) + sb_a * sb_a) + y3) + 2.0 * a;
  tb_a = y3 + 2.0 * a;
  ub_a = y3 + 2.0 * a;
  vb_a = y3 + 2.0 * a;
  wb_a = y3 + 2.0 * a;
  xb_a = y3 + 2.0 * a;
  yb_a = y3 + 2.0 * a;
  ac_a = y3 + 2.0 * a;
  bc_a = y3 + 2.0 * a;
  cc_a = y3 + 2.0 * a;
  dc_a = y3 + 2.0 * a;
  ec_a = y3 + 2.0 * a;
  fc_a = y3 + 2.0 * a;
  gc_a = y3 + 2.0 * a;
  hc_a = y3 + 2.0 * a;
  ic_a = y3 + 2.0 * a;
  jc_a = y3 + 2.0 * a;
  jc_a = (sqrt((b_y1 * b_y1 + y2 * y2) + jc_a * jc_a) + y3) + 2.0 * a;
  kc_a = y3 + 2.0 * a;
  lc_a = y3 + 2.0 * a;
  lc_a = (b_y1 * b_y1 + y2 * y2) + lc_a * lc_a;
  mc_a = y3 + 2.0 * a;
  nc_a = y3 + 2.0 * a;
  oc_a = y3 + 2.0 * a;
  pc_a = y3 + 2.0 * a;
  pc_a = (sqrt((b_y1 * b_y1 + y2 * y2) + pc_a * pc_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  c_x = cos(b);
  qc_a = y3 + 2.0 * a;
  rc_a = y3 + 2.0 * a;
  sc_a = y3 + 2.0 * a;
  tc_a = y3 + 2.0 * a;
  d_x = cos(b);
  uc_a = y3 + 2.0 * a;
  vc_a = y3 + 2.0 * a;
  wc_a = y3 + 2.0 * a;
  xc_a = y3 + 2.0 * a;
  yc_a = y3 + 2.0 * a;
  ad_a = y3 + 2.0 * a;
  bd_a = y3 + 2.0 * a;
  cd_a = y3 + 2.0 * a;
  e_x = cos(b);
  dd_a = y3 + 2.0 * a;
  ed_a = y3 + 2.0 * a;
  fd_a = y3 + 2.0 * a;
  gd_a = y3 + 2.0 * a;
  gd_a = (sqrt((b_y1 * b_y1 + y2 * y2) + gd_a * gd_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  f_x = cos(b);
  hd_a = y3 + 2.0 * a;
  id_a = y3 + 2.0 * a;
  jd_a = y3 + 2.0 * a;
  kd_a = y3 + 2.0 * a;
  g_x = cos(b);
  ld_a = y3 + 2.0 * a;
  md_a = y3 + 2.0 * a;
  nd_a = y3 + 2.0 * a;
  od_a = b_y1 * cos(b) - y3 * sin(b);
  pd_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  h_x = sin(b);
  qd_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  rd_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  sd_a = y3 + 2.0 * a;
  td_a = y3 + 2.0 * a;
  ud_a = y3 + 2.0 * a;
  vd_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  wd_a = y3 + 2.0 * a;
  i_x = sin(b);
  xd_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  yd_a = y3 + 2.0 * a;
  ae_a = y3 + 2.0 * a;
  be_a = sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3;
  ce_a = y3 + 2.0 * a;
  de_a = y3 + 2.0 * a;
  ee_a = y3 + 2.0 * a;
  fe_a = y3 + 2.0 * a;
  fe_a = (sqrt((b_y1 * b_y1 + y2 * y2) + fe_a * fe_a) + y3) + 2.0 * a;
  ge_a = y3 + 2.0 * a;
  he_a = y3 + 2.0 * a;
  ie_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  je_a = y3 + 2.0 * a;
  ke_a = y3 + 2.0 * a;
  le_a = y3 + 2.0 * a;
  me_a = y3 + 2.0 * a;
  me_a = (sqrt((b_y1 * b_y1 + y2 * y2) + me_a * me_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  ne_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  oe_a = y3 + 2.0 * a;
  pe_a = y3 + 2.0 * a;
  qe_a = y3 + 2.0 * a;
  re_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  se_a = y3 + 2.0 * a;
  j_x = sin(b);
  te_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  k_x = 1.0 / tan(b);
  ue_a = y3 + 2.0 * a;
  ve_a = y3 + 2.0 * a;
  we_a = y3 + 2.0 * a;
  xe_a = y3 + 2.0 * a;
  ye_a = y3 + 2.0 * a;
  ye_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ye_a * ye_a) + y3) + 2.0 * a;
  af_a = y3 + 2.0 * a;
  bf_a = y3 + 2.0 * a;
  cf_a = y3 + 2.0 * a;
  df_a = y3 + 2.0 * a;
  ef_a = y3 + 2.0 * a;
  ff_a = y3 + 2.0 * a;
  gf_a = y3 + 2.0 * a;
  gf_a = (sqrt((b_y1 * b_y1 + y2 * y2) + gf_a * gf_a) + y3) + 2.0 * a;
  hf_a = y3 + 2.0 * a;
  if_a = y3 + 2.0 * a;
  jf_a = y3 + 2.0 * a;
  kf_a = y3 + 2.0 * a;
  lf_a = y3 + 2.0 * a;
  mf_a = y3 + 2.0 * a;
  nf_a = y3 + 2.0 * a;
  nf_a = (sqrt((b_y1 * b_y1 + y2 * y2) + nf_a * nf_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  of_a = y3 + 2.0 * a;
  pf_a = y3 + 2.0 * a;
  qf_a = y3 + 2.0 * a;
  rf_a = y3 + 2.0 * a;
  sf_a = y3 + 2.0 * a;
  tf_a = y3 + 2.0 * a;
  uf_a = y3 + 2.0 * a;
  vf_a = y3 + 2.0 * a;
  wf_a = y3 + 2.0 * a;
  xf_a = y3 + 2.0 * a;
  yf_a = y3 + 2.0 * a;
  ag_a = y3 + 2.0 * a;
  bg_a = y3 + 2.0 * a;
  cg_a = y3 + 2.0 * a;
  dg_a = y3 + 2.0 * a;
  eg_a = y3 + 2.0 * a;
  fg_a = y3 + 2.0 * a;
  gg_a = y3 + 2.0 * a;
  hg_a = y3 + 2.0 * a;
  ig_a = y3 + 2.0 * a;
  ig_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ig_a * ig_a) + y3) + 2.0 * a;
  jg_a = y3 + 2.0 * a;
  kg_a = y3 + 2.0 * a;
  lg_a = y3 + 2.0 * a;
  mg_a = y3 + 2.0 * a;
  ng_a = y3 + 2.0 * a;
  og_a = y3 + 2.0 * a;
  pg_a = y3 + 2.0 * a;
  pg_a = (sqrt((b_y1 * b_y1 + y2 * y2) + pg_a * pg_a) + y3) + 2.0 * a;
  qg_a = y3 + 2.0 * a;
  rg_a = y3 + 2.0 * a;
  sg_a = y3 + 2.0 * a;
  tg_a = y3 + 2.0 * a;
  ug_a = y3 + 2.0 * a;
  vg_a = y3 + 2.0 * a;
  wg_a = y3 + 2.0 * a;
  wg_a = (sqrt((b_y1 * b_y1 + y2 * y2) + wg_a * wg_a) + y3) + 2.0 * a;
  xg_a = y3 + 2.0 * a;
  yg_a = y3 + 2.0 * a;
  ah_a = y3 + 2.0 * a;
  bh_a = y3 + 2.0 * a;
  ch_a = y3 + 2.0 * a;
  dh_a = y3 + 2.0 * a;
  eh_a = y3 + 2.0 * a;
  fh_a = y3 + 2.0 * a;
  gh_a = y3 + 2.0 * a;
  hh_a = y3 + 2.0 * a;
  ih_a = y3 + 2.0 * a;
  jh_a = y3 + 2.0 * a;
  kh_a = y3 + 2.0 * a;
  lh_a = y3 + 2.0 * a;
  mh_a = y3 + 2.0 * a;
  mh_a = (sqrt((b_y1 * b_y1 + y2 * y2) + mh_a * mh_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  nh_a = y3 + 2.0 * a;
  oh_a = y3 + 2.0 * a;
  ph_a = y3 + 2.0 * a;
  qh_a = y3 + 2.0 * a;
  rh_a = y3 + 2.0 * a;
  sh_a = y3 + 2.0 * a;
  th_a = y3 + 2.0 * a;
  uh_a = y3 + 2.0 * a;
  vh_a = y3 + 2.0 * a;
  wh_a = y3 + 2.0 * a;
  xh_a = y3 + 2.0 * a;
  xh_a = (sqrt((b_y1 * b_y1 + y2 * y2) + xh_a * xh_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  yh_a = y3 + 2.0 * a;
  ai_a = y3 + 2.0 * a;
  bi_a = y3 + 2.0 * a;
  ci_a = y3 + 2.0 * a;
  di_a = y3 + 2.0 * a;
  ei_a = y3 + 2.0 * a;
  ei_a = (b_y1 * b_y1 + y2 * y2) + ei_a * ei_a;
  fi_a = y3 + 2.0 * a;
  gi_a = y3 + 2.0 * a;
  hi_a = y3 + 2.0 * a;
  ii_a = y3 + 2.0 * a;
  ji_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  ki_a = y3 + 2.0 * a;
  li_a = y3 + 2.0 * a;
  mi_a = y3 + 2.0 * a;
  ni_a = y3 + 2.0 * a;
  ni_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ni_a * ni_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  oi_a = y3 + 2.0 * a;
  pi_a = y3 + 2.0 * a;
  qi_a = y3 + 2.0 * a;
  qi_a = (sqrt((b_y1 * b_y1 + y2 * y2) + qi_a * qi_a) + y3) + 2.0 * a;
  ri_a = y3 + 2.0 * a;
  si_a = y3 + 2.0 * a;
  ti_a = y3 + 2.0 * a;
  ui_a = y3 + 2.0 * a;
  vi_a = y3 + 2.0 * a;
  vi_a = (sqrt((b_y1 * b_y1 + y2 * y2) + vi_a * vi_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  wi_a = y3 + 2.0 * a;
  xi_a = y3 + 2.0 * a;
  yi_a = y3 + 2.0 * a;
  aj_a = y3 + 2.0 * a;
  bj_a = y3 + 2.0 * a;
  cj_a = y3 + 2.0 * a;
  dj_a = y3 + 2.0 * a;
  ej_a = y3 + 2.0 * a;
  fj_a = y3 + 2.0 * a;
  fj_a = (b_y1 * b_y1 + y2 * y2) + fj_a * fj_a;
  gj_a = y3 + 2.0 * a;
  gj_a = (sqrt((b_y1 * b_y1 + y2 * y2) + gj_a * gj_a) + y3) + 2.0 * a;
  hj_a = y3 + 2.0 * a;
  ij_a = y3 + 2.0 * a;
  ij_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ij_a * ij_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  jj_a = y3 + 2.0 * a;
  kj_a = y3 + 2.0 * a;
  lj_a = y3 + 2.0 * a;
  mj_a = y3 + 2.0 * a;
  nj_a = y3 + 2.0 * a;
  oj_a = y3 + 2.0 * a;
  pj_a = y3 + 2.0 * a;
  qj_a = y3 + 2.0 * a;
  rj_a = y3 + 2.0 * a;
  sj_a = y3 + 2.0 * a;
  tj_a = y3 + 2.0 * a;
  uj_a = y3 + 2.0 * a;
  vj_a = y3 + 2.0 * a;
  wj_a = y3 + 2.0 * a;
  xj_a = y3 + 2.0 * a;
  yj_a = y3 + 2.0 * a;
  ak_a = y3 + 2.0 * a;
  bk_a = y3 + 2.0 * a;
  ck_a = y3 + 2.0 * a;
  ck_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ck_a * ck_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  dk_a = y3 + 2.0 * a;
  ek_a = y3 + 2.0 * a;
  fk_a = y3 + 2.0 * a;
  gk_a = y3 + 2.0 * a;
  hk_a = y3 + 2.0 * a;
  ik_a = y3 + 2.0 * a;
  jk_a = y3 + 2.0 * a;
  *e22 = (B1 * (0.125 * (((1.0 - 2.0 * nu) * ((1.0 / sqrt((b_y1 * b_y1 + y2 * y2)
    + y3 * y3) * y2 / (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3) + 1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + b_a * b_a) * y2 / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + c_a * c_a) + y3) + 2.0 * a)) - cos(b) * (1.0 / sqrt((b_y1 * b_y1 + y2 *
    y2) + y3 * y3) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin
    (b)) - y3 * cos(b)) + 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + d_a * d_a) * y2 /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + e_a * e_a) - b_y1 * sin(b)) + (y3 + 2.0 * a)
     * cos(b)))) - 2.0 * y2 * ((1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) /
    (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3) + 1.0 / sqrt((b_y1 * b_y1 +
    y2 * y2) + f_a * f_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + g_a * g_a) + y3) +
    2.0 * a)) - cos(b) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) + 1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + h_a * h_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    i_a * i_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))))) - y2 * y2 *
            ((((-1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) /
                (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3) * y2 - 1.0 /
                ((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (j_a * j_a) * y2) - 1.0 /
               powd_snf((b_y1 * b_y1 + y2 * y2) + k_a * k_a, 1.5) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + l_a * l_a) + y3) + 2.0 * a) * y2) - 1.0 / ((b_y1 *
    b_y1 + y2 * y2) + m_a * m_a) / (n_a * n_a) * y2) - cos(b) * (((-1.0 /
    powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) * y2 - 1.0 / ((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) / (o_a * o_a) * y2) - 1.0 / powd_snf((b_y1 *
    b_y1 + y2 * y2) + p_a * p_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + q_a *
    q_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * y2) - 1.0 / ((b_y1 * b_y1
    + y2 * y2) + r_a * r_a) / (s_a * s_a) * y2))) / M_PI / (1.0 -
            nu) + 0.25 * ((((((((((1.0 - 2.0 * nu) * (((2.0 - 2.0 * nu) * (x * x)
    - nu) / sqrt((b_y1 * b_y1 + y2 * y2) + t_a * t_a) * y2 / ((sqrt((b_y1 * b_y1
    + y2 * y2) + u_a * u_a) + y3) + 2.0 * a) - (((2.0 - 2.0 * nu) * (b_x * b_x)
    + 1.0) - 2.0 * nu) * cos(b) / sqrt((b_y1 * b_y1 + y2 * y2) + v_a * v_a) * y2
    / ((sqrt((b_y1 * b_y1 + y2 * y2) + w_a * w_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b))) + (1.0 - 2.0 * nu) / (x_a * x_a) * (((b_y1 * (1.0 / tan(b)) *
    ((1.0 - 2.0 * nu) - a / sqrt((b_y1 * b_y1 + y2 * y2) + y_a * y_a)) + nu *
    (y3 + 2.0 * a)) - a) + y2 * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + ab_a *
    ab_a) + y3) + 2.0 * a) * (nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + bb_a *
    bb_a))) / sqrt((b_y1 * b_y1 + y2 * y2) + cb_a * cb_a) * y2) - (1.0 - 2.0 *
    nu) / ((sqrt((b_y1 * b_y1 + y2 * y2) + db_a * db_a) + y3) + 2.0 * a) * (((a *
    b_y1 * (1.0 / tan(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + eb_a * eb_a,
    1.5) * y2 + 2.0 * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + fb_a * fb_a) + y3) +
    2.0 * a) * (nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + gb_a * gb_a))) -
    powd_snf(y2, 3.0) / (hb_a * hb_a) * (nu + a / sqrt((b_y1 * b_y1 + y2 * y2)
    + ib_a * ib_a)) / sqrt((b_y1 * b_y1 + y2 * y2) + jb_a * jb_a)) - powd_snf
    (y2, 3.0) / ((sqrt((b_y1 * b_y1 + y2 * y2) + kb_a * kb_a) + y3) + 2.0 * a) *
    a / powd_snf((b_y1 * b_y1 + y2 * y2) + lb_a * lb_a, 1.5))) + (1.0 - 2.0 *
    nu) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) / (mb_a *
    mb_a) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + nb_a * nb_a)) / sqrt
    ((b_y1 * b_y1 + y2 * y2) + ob_a * ob_a) * y2) + (1.0 - 2.0 * nu) * (b_y1 *
    cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + pb_a * pb_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * a /
    powd_snf((b_y1 * b_y1 + y2 * y2) + qb_a * qb_a, 1.5) * y2) + 3.0 * a * y2
    * (y3 + a) * (1.0 / tan(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + rb_a *
    rb_a, 2.5) * b_y1) - (y3 + a) / (sb_a * sb_a) * (((-2.0 * nu + 1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + tb_a * tb_a) * ((1.0 - 2.0 * nu) * b_y1 * (1.0 /
    tan(b)) - a)) + y2 * y2 / sqrt((b_y1 * b_y1 + y2 * y2) + ub_a * ub_a) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + vb_a * vb_a) + y3) + 2.0 * a) * (2.0 * nu +
    a / sqrt((b_y1 * b_y1 + y2 * y2) + wb_a * wb_a))) + a * (y2 * y2) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + xb_a * xb_a, 1.5)) / sqrt((b_y1 * b_y1
    + y2 * y2) + yb_a * yb_a) * y2) + (y3 + a) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + ac_a * ac_a) + y3) + 2.0 * a) * ((((((-1.0 / powd_snf((b_y1 * b_y1 + y2
    * y2) + bc_a * bc_a, 1.5) * ((1.0 - 2.0 * nu) * b_y1 * (1.0 / tan(b)) - a) *
    y2 + 2.0 * y2 / sqrt((b_y1 * b_y1 + y2 * y2) + cc_a * cc_a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + dc_a * dc_a) + y3) + 2.0 * a) * (2.0 * nu + a / sqrt((b_y1
    * b_y1 + y2 * y2) + ec_a * ec_a))) - powd_snf(y2, 3.0) / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + fc_a * fc_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + gc_a * gc_a) + y3) + 2.0 * a) * (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 *
    y2) + hc_a * hc_a))) - powd_snf(y2, 3.0) / ((b_y1 * b_y1 + y2 * y2) +
    ic_a * ic_a) / (jc_a * jc_a) * (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 * y2)
    + kc_a * kc_a))) - powd_snf(y2, 3.0) / (lc_a * lc_a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + mc_a * mc_a) + y3) + 2.0 * a) * a) + 2.0 * a / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + nc_a * nc_a, 1.5) * y2) - 3.0 * a * powd_snf
    (y2, 3.0) / powd_snf((b_y1 * b_y1 + y2 * y2) + oc_a * oc_a, 2.5))) - (y3
              + a) / (pc_a * pc_a) * (((c_x * c_x - 1.0 / sqrt((b_y1 * b_y1 + y2
    * y2) + qc_a * qc_a) * ((1.0 - 2.0 * nu) * (b_y1 * cos(b) + (y3 + 2.0 * a) *
    sin(b)) * (1.0 / tan(b)) + a * cos(b))) + a * (y3 + 2.0 * a) * (b_y1 * cos(b)
    + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) / powd_snf((b_y1 * b_y1 + y2 *
    y2) + rc_a * rc_a, 1.5)) - 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + sc_a * sc_a)
              / ((sqrt((b_y1 * b_y1 + y2 * y2) + tc_a * tc_a) - b_y1 * sin(b)) +
                 (y3 + 2.0 * a) * cos(b)) * (y2 * y2 * (d_x * d_x) - a * (b_y1 *
    cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) / sqrt((b_y1 * b_y1 + y2 *
    y2) + uc_a * uc_a) * ((sqrt((b_y1 * b_y1 + y2 * y2) + vc_a * vc_a) * cos(b)
    + y3) + 2.0 * a))) / sqrt((b_y1 * b_y1 + y2 * y2) + wc_a * wc_a) * y2) + (y3
             + a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + xc_a * xc_a) - b_y1 * sin(b))
                     + (y3 + 2.0 * a) * cos(b)) * ((((1.0 / powd_snf((b_y1 *
    b_y1 + y2 * y2) + yc_a * yc_a, 1.5) * ((1.0 - 2.0 * nu) * (b_y1 * cos(b) +
    (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) + a * cos(b)) * y2 - 3.0 * a * (y3
    + 2.0 * a) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + ad_a * ad_a, 2.5) * y2) + 1.0 /
    powd_snf((b_y1 * b_y1 + y2 * y2) + bd_a * bd_a, 1.5) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + cd_a * cd_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    (y2 * y2 * (e_x * e_x) - a * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) *
     (1.0 / tan(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + dd_a * dd_a) * ((sqrt((b_y1
    * b_y1 + y2 * y2) + ed_a * ed_a) * cos(b) + y3) + 2.0 * a)) * y2) + 1.0 /
              ((b_y1 * b_y1 + y2 * y2) + fd_a * fd_a) / (gd_a * gd_a) * (y2 * y2
    * (f_x * f_x) - a * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b))
    / sqrt((b_y1 * b_y1 + y2 * y2) + hd_a * hd_a) * ((sqrt((b_y1 * b_y1 + y2 *
    y2) + id_a * id_a) * cos(b) + y3) + 2.0 * a)) * y2) - 1.0 / sqrt((b_y1 *
    b_y1 + y2 * y2) + jd_a * jd_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + kd_a *
    kd_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((2.0 * y2 * (g_x * g_x)
    + a * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + ld_a * ld_a, 1.5) * ((sqrt((b_y1 *
    b_y1 + y2 * y2) + md_a * md_a) * cos(b) + y3) + 2.0 * a) * y2) - a * (b_y1 *
    cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) / ((b_y1 * b_y1 + y2 * y2)
    + nd_a * nd_a) * cos(b) * y2))) / M_PI / (1.0 - nu)) + B2 *
          (0.125 * ((((((2.0 - 2.0 * nu) * ((((-2.0 / b_y1 / (1.0 + y2 * y2 /
    (b_y1 * b_y1)) + 1.0 / (b_y1 * cos(b) - y3 * sin(b)) / (1.0 + y2 * y2 /
    (od_a * od_a))) + ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) / (b_y1
    * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b)) + y2 * y2 / sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) * sin(b) / (b_y1 * (b_y1 * cos(b) - y3 * sin(b))
    + y2 * y2 * cos(b))) - 2.0 * (y2 * y2) * sqrt((b_y1 * b_y1 + y2 * y2) + y3 *
    y3) * sin(b) / (pd_a * pd_a) * cos(b)) / (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2
    * y2) + y3 * y3) * (h_x * h_x) / (qd_a * qd_a))) + 1.0 / (b_y1 * cos(b) +
    (y3 + 2.0 * a) * sin(b)) / (1.0 + y2 * y2 / (rd_a * rd_a))) + ((sqrt((b_y1 *
    b_y1 + y2 * y2) + sd_a * sd_a) * sin(b) / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0
    * a) * sin(b)) + y2 * y2 * cos(b)) + y2 * y2 / sqrt((b_y1 * b_y1 + y2 * y2)
    + td_a * td_a) * sin(b) / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b))
    + y2 * y2 * cos(b))) - 2.0 * (y2 * y2) * sqrt((b_y1 * b_y1 + y2 * y2) + ud_a
    * ud_a) * sin(b) / (vd_a * vd_a) * cos(b)) / (1.0 + y2 * y2 * ((b_y1 * b_y1
    + y2 * y2) + wd_a * wd_a) * (i_x * i_x) / (xd_a * xd_a))) + b_y1 * (1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (sqrt((b_y1 * b_y1 + y2 * y2) + y3
    * y3) - y3) + 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + yd_a * yd_a) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + ae_a * ae_a) + y3) + 2.0 * a))) + b_y1 * y2 * (((
    -1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / (sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) - y3) * y2 - 1.0 / ((b_y1 * b_y1 + y2 * y2) + y3 *
    y3) / (be_a * be_a) * y2) - 1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + ce_a
    * ce_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + de_a * de_a) + y3) + 2.0 * a)
    * y2) - 1.0 / ((b_y1 * b_y1 + y2 * y2) + ee_a * ee_a) / (fe_a * fe_a) * y2))
                      - (b_y1 * cos(b) - y3 * sin(b)) / sqrt((b_y1 * b_y1 + y2 *
    y2) + y3 * y3) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b))
                      - y3 * cos(b))) - (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b))
                     / sqrt((b_y1 * b_y1 + y2 * y2) + ge_a * ge_a) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + he_a * he_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
              cos(b))) - y2 * (((-(b_y1 * cos(b) - y3 * sin(b)) / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) * y2 - (b_y1 * cos(b) - y3 * sin(b))
    / ((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (ie_a * ie_a) * y2) - (b_y1 * cos(b)
    + (y3 + 2.0 * a) * sin(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + je_a *
    je_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ke_a * ke_a) - b_y1 * sin(b))
                  + (y3 + 2.0 * a) * cos(b)) * y2) - (b_y1 * cos(b) + (y3 + 2.0 *
    a) * sin(b)) / ((b_y1 * b_y1 + y2 * y2) + le_a * le_a) / (me_a * me_a) * y2))
           / M_PI / (1.0 - nu) + 0.25 * (((((((((((((((((2.0 - 2.0
    * nu) * (1.0 - 2.0 * nu) * ((-1.0 / b_y1 / (1.0 + y2 * y2 / (b_y1 * b_y1)) +
    1.0 / (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / (1.0 + y2 * y2 / (ne_a *
    ne_a))) + ((sqrt((b_y1 * b_y1 + y2 * y2) + oe_a * oe_a) * sin(b) / (b_y1 *
    (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b)) + y2 * y2 /
    sqrt((b_y1 * b_y1 + y2 * y2) + pe_a * pe_a) * sin(b) / (b_y1 * (b_y1 * cos(b)
    + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b))) - 2.0 * (y2 * y2) * sqrt
    ((b_y1 * b_y1 + y2 * y2) + qe_a * qe_a) * sin(b) / (re_a * re_a) * cos(b)) /
    (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2) + se_a * se_a) * (j_x * j_x) /
    (te_a * te_a))) * (k_x * k_x) + (1.0 - 2.0 * nu) / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + ue_a * ue_a) + y3) + 2.0 * a) * (((-1.0 + 2.0 * nu) + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + ve_a * ve_a)) * (1.0 / tan(b)) + b_y1 / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + we_a * we_a) + y3) + 2.0 * a) * (nu + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + xe_a * xe_a)))) - (1.0 - 2.0 * nu) * (y2 * y2) / (ye_a *
    ye_a) * (((-1.0 + 2.0 * nu) + a / sqrt((b_y1 * b_y1 + y2 * y2) + af_a * af_a))
             * (1.0 / tan(b)) + b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + bf_a *
    bf_a) + y3) + 2.0 * a) * (nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + cf_a *
    cf_a))) / sqrt((b_y1 * b_y1 + y2 * y2) + df_a * df_a)) + (1.0 - 2.0 * nu) *
    y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + ef_a * ef_a) + y3) + 2.0 * a) * ((-a /
    powd_snf((b_y1 * b_y1 + y2 * y2) + ff_a * ff_a, 1.5) * y2 * (1.0 / tan(b))
    - b_y1 / (gf_a * gf_a) * (nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + hf_a *
    hf_a)) / sqrt((b_y1 * b_y1 + y2 * y2) + if_a * if_a) * y2) - y2 / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + jf_a * jf_a) + y3) + 2.0 * a) * a / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + kf_a * kf_a, 1.5) * b_y1)) - (1.0 - 2.0 * nu) *
    (1.0 / tan(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + lf_a * lf_a) - b_y1 * sin
                       (b)) + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + mf_a * mf_a) / cos(b))) + (1.0 - 2.0 * nu) * (y2 * y2) *
    (1.0 / tan(b)) / (nf_a * nf_a) * (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2) +
    of_a * of_a) / cos(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + pf_a * pf_a)) + (1.0
    - 2.0 * nu) * (y2 * y2) * (1.0 / tan(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    qf_a * qf_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + rf_a * rf_a, 1.5) / cos(b)) - a * (y3 + a) * (1.0
    / tan(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + sf_a * sf_a, 1.5)) + 3.0 *
    a * (y2 * y2) * (y3 + a) * (1.0 / tan(b)) / powd_snf((b_y1 * b_y1 + y2 *
    y2) + tf_a * tf_a, 2.5)) + (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) + uf_a *
    uf_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + vf_a * vf_a) + y3) + 2.0 * a) *
    (((1.0 - 2.0 * nu) * (1.0 / tan(b)) - 2.0 * nu * b_y1 / ((sqrt((b_y1 * b_y1
    + y2 * y2) + wf_a * wf_a) + y3) + 2.0 * a)) - a * b_y1 / sqrt((b_y1 * b_y1 +
    y2 * y2) + xf_a * xf_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + yf_a * yf_a)
    + 1.0 / ((sqrt((b_y1 * b_y1 + y2 * y2) + ag_a * ag_a) + y3) + 2.0 * a)))) -
    y2 * y2 * (y3 + a) / powd_snf((b_y1 * b_y1 + y2 * y2) + bg_a * bg_a, 1.5)
    / ((sqrt((b_y1 * b_y1 + y2 * y2) + cg_a * cg_a) + y3) + 2.0 * a) * (((1.0 -
    2.0 * nu) * (1.0 / tan(b)) - 2.0 * nu * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + dg_a * dg_a) + y3) + 2.0 * a)) - a * b_y1 / sqrt((b_y1 * b_y1 + y2 * y2) +
    eg_a * eg_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + fg_a * fg_a) + 1.0 /
                    ((sqrt((b_y1 * b_y1 + y2 * y2) + gg_a * gg_a) + y3) + 2.0 *
                     a)))) - y2 * y2 * (y3 + a) / ((b_y1 * b_y1 + y2 * y2) +
    hg_a * hg_a) / (ig_a * ig_a) * (((1.0 - 2.0 * nu) * (1.0 / tan(b)) - 2.0 *
    nu * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + jg_a * jg_a) + y3) + 2.0 * a))
    - a * b_y1 / sqrt((b_y1 * b_y1 + y2 * y2) + kg_a * kg_a) * (1.0 / sqrt((b_y1
    * b_y1 + y2 * y2) + lg_a * lg_a) + 1.0 / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    mg_a * mg_a) + y3) + 2.0 * a)))) + y2 * (y3 + a) / sqrt((b_y1 * b_y1 + y2 *
    y2) + ng_a * ng_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + og_a * og_a) + y3) +
    2.0 * a) * ((2.0 * nu * b_y1 / (pg_a * pg_a) / sqrt((b_y1 * b_y1 + y2 * y2)
    + qg_a * qg_a) * y2 + a * b_y1 / powd_snf((b_y1 * b_y1 + y2 * y2) + rg_a *
    rg_a, 1.5) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + sg_a * sg_a) + 1.0 /
                  ((sqrt((b_y1 * b_y1 + y2 * y2) + tg_a * tg_a) + y3) + 2.0 * a))
                 * y2) - a * b_y1 / sqrt((b_y1 * b_y1 + y2 * y2) + ug_a * ug_a) *
                (-1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + vg_a * vg_a, 1.5) *
                 y2 - 1.0 / (wg_a * wg_a) / sqrt((b_y1 * b_y1 + y2 * y2) + xg_a *
    xg_a) * y2))) + (y3 + a) * (1.0 / tan(b)) / sqrt((b_y1 * b_y1 + y2 * y2) +
    yg_a * yg_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ah_a * ah_a) - b_y1 * sin(b))
                    + (y3 + 2.0 * a) * cos(b)) * (((-2.0 + 2.0 * nu) * cos(b) +
    ((sqrt((b_y1 * b_y1 + y2 * y2) + bh_a * bh_a) * cos(b) + y3) + 2.0 * a) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + ch_a * ch_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)) * (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2) + dh_a * dh_a) / cos
                    (b))) + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) + eh_a
    * eh_a) / cos(b))) - y2 * y2 * (y3 + a) * (1.0 / tan(b)) / powd_snf((b_y1
    * b_y1 + y2 * y2) + fh_a * fh_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    gh_a * gh_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((-2.0 + 2.0 *
    nu) * cos(b) + ((sqrt((b_y1 * b_y1 + y2 * y2) + hh_a * hh_a) * cos(b) + y3)
                    + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ih_a * ih_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt((b_y1 * b_y1 +
    y2 * y2) + jh_a * jh_a) / cos(b))) + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2
    * y2) + kh_a * kh_a) / cos(b))) - y2 * y2 * (y3 + a) * (1.0 / tan(b)) /
             ((b_y1 * b_y1 + y2 * y2) + lh_a * lh_a) / (mh_a * mh_a) * (((-2.0 +
    2.0 * nu) * cos(b) + ((sqrt((b_y1 * b_y1 + y2 * y2) + nh_a * nh_a) * cos(b)
    + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + oh_a * oh_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2)
    + ph_a * ph_a) / cos(b))) + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) +
    qh_a * qh_a) / cos(b))) + y2 * (y3 + a) * (1.0 / tan(b)) / sqrt((b_y1 * b_y1
              + y2 * y2) + rh_a * rh_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + sh_a *
    sh_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((1.0 / sqrt((b_y1 *
    b_y1 + y2 * y2) + th_a * th_a) * cos(b) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + uh_a * uh_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt
    ((b_y1 * b_y1 + y2 * y2) + vh_a * vh_a) / cos(b)) - ((sqrt((b_y1 * b_y1 + y2
    * y2) + wh_a * wh_a) * cos(b) + y3) + 2.0 * a) / (xh_a * xh_a) * (1.0 + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + yh_a * yh_a) / cos(b)) / sqrt((b_y1 * b_y1 +
    y2 * y2) + ai_a * ai_a) * y2) - ((sqrt((b_y1 * b_y1 + y2 * y2) + bi_a * bi_a)
    * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ci_a * ci_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 * b_y1 +
    y2 * y2) + di_a * di_a, 1.5) / cos(b) * y2) - 2.0 * a * (y3 + 2.0 * a) /
             (ei_a * ei_a) / cos(b) * y2)) / M_PI / (1.0 - nu))) +
    B3 * (0.125 * (((1.0 - 2.0 * nu) * sin(b) * (1.0 / sqrt((b_y1 * b_y1 + y2 *
    y2) + y3 * y3) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin
    (b)) - y3 * cos(b)) + 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + fi_a * fi_a) * y2
             / ((sqrt((b_y1 * b_y1 + y2 * y2) + gi_a * gi_a) - b_y1 * sin(b)) +
                (y3 + 2.0 * a) * cos(b))) - 2.0 * y2 * sin(b) * (1.0 / sqrt
             ((b_y1 * b_y1 + y2 * y2) + y3 * y3) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) + 1.0 / sqrt((b_y1 * b_y1 + y2 *
    y2) + hi_a * hi_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ii_a * ii_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)))) - y2 * y2 * sin(b) * (((-1.0 /
              powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / ((sqrt((b_y1
    * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) * y2 - 1.0 /
              ((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (ji_a * ji_a) * y2) - 1.0 /
             powd_snf((b_y1 * b_y1 + y2 * y2) + ki_a * ki_a, 1.5) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + li_a * li_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
              cos(b)) * y2) - 1.0 / ((b_y1 * b_y1 + y2 * y2) + mi_a * mi_a) /
            (ni_a * ni_a) * y2)) / M_PI / (1.0 - nu) + 0.25 *
          (((((1.0 - 2.0 * nu) * ((((-sin(b) / sqrt((b_y1 * b_y1 + y2 * y2) +
    oi_a * oi_a) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + pi_a * pi_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) + y2 / (qi_a * qi_a) * (1.0 + a / sqrt
    ((b_y1 * b_y1 + y2 * y2) + ri_a * ri_a)) / sqrt((b_y1 * b_y1 + y2 * y2) +
    si_a * si_a) * b_y1) + y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + ti_a * ti_a) +
    y3) + 2.0 * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + ui_a * ui_a, 1.5)
    * b_y1) - (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / (vi_a * vi_a) * (cos(b)
    + a / sqrt((b_y1 * b_y1 + y2 * y2) + wi_a * wi_a)) / sqrt((b_y1 * b_y1 + y2 *
    y2) + xi_a * xi_a) * y2) - (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + yi_a * yi_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + aj_a * aj_a, 1.5) *
    y2) - y2 * (y3 + a) / powd_snf((b_y1 * b_y1 + y2 * y2) + bj_a * bj_a, 1.5)
              * (a / ((b_y1 * b_y1 + y2 * y2) + cj_a * cj_a) + 1.0 / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + dj_a * dj_a) + y3) + 2.0 * a)) * b_y1) + b_y1 *
             (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) + ej_a * ej_a) * (-2.0 * a /
              (fj_a * fj_a) * y2 - 1.0 / (gj_a * gj_a) / sqrt((b_y1 * b_y1 + y2 *
    y2) + hj_a * hj_a) * y2)) + (y3 + a) / (ij_a * ij_a) * ((sin(b) * (cos(b) -
    a / sqrt((b_y1 * b_y1 + y2 * y2) + jj_a * jj_a)) + (b_y1 * cos(b) + (y3 +
    2.0 * a) * sin(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + kj_a * kj_a) * (1.0 + a *
    (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) + lj_a * lj_a))) - 1.0 / sqrt
             ((b_y1 * b_y1 + y2 * y2) + mj_a * mj_a) / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + nj_a * nj_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (y2 * y2 *
              cos(b) * sin(b) - a * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) /
              sqrt((b_y1 * b_y1 + y2 * y2) + oj_a * oj_a) * ((sqrt((b_y1 * b_y1
    + y2 * y2) + pj_a * pj_a) * cos(b) + y3) + 2.0 * a))) / sqrt((b_y1 * b_y1 +
              y2 * y2) + qj_a * qj_a) * y2) - (y3 + a) / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + rj_a * rj_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
           (((((sin(b) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + sj_a * sj_a,
    1.5) * y2 - (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / powd_snf((b_y1 *
    b_y1 + y2 * y2) + tj_a * tj_a, 1.5) * (1.0 + a * (y3 + 2.0 * a) / ((b_y1 *
    b_y1 + y2 * y2) + uj_a * uj_a)) * y2) - 2.0 * (b_y1 * cos(b) + (y3 + 2.0 * a)
    * sin(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + vj_a * vj_a, 2.5) * a *
               (y3 + 2.0 * a) * y2) + 1.0 / powd_snf((b_y1 * b_y1 + y2 * y2)
    + wj_a * wj_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + xj_a * xj_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * (y2 * y2 * cos(b) * sin(b) - a * (b_y1 *
    cos(b) + (y3 + 2.0 * a) * sin(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + yj_a *
    yj_a) * ((sqrt((b_y1 * b_y1 + y2 * y2) + ak_a * ak_a) * cos(b) + y3) + 2.0 *
             a)) * y2) + 1.0 / ((b_y1 * b_y1 + y2 * y2) + bk_a * bk_a) / (ck_a *
              ck_a) * (y2 * y2 * cos(b) * sin(b) - a * (b_y1 * cos(b) + (y3 +
    2.0 * a) * sin(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + dk_a * dk_a) * ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + ek_a * ek_a) * cos(b) + y3) + 2.0 * a)) * y2) -
            1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + fk_a * fk_a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + gk_a * gk_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
            ((2.0 * y2 * cos(b) * sin(b) + a * (b_y1 * cos(b) + (y3 + 2.0 * a) *
    sin(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + hk_a * hk_a, 1.5) * ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + ik_a * ik_a) * cos(b) + y3) + 2.0 * a) * y2) - a *
             (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / ((b_y1 * b_y1 + y2 * y2)
              + jk_a * jk_a) * cos(b) * y2))) / M_PI / (1.0 - nu));
  b_a = y3 + 2.0 * a;
  c_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  d_a = y3 + 2.0 * a;
  e_a = y3 + 2.0 * a;
  f_a = y3 + 2.0 * a;
  g_a = y3 + 2.0 * a;
  h_a = y3 + 2.0 * a;
  i_a = y3 + 2.0 * a;
  j_a = y3 + 2.0 * a;
  k_a = y3 + 2.0 * a;
  l_a = y3 + 2.0 * a;
  l_a = (sqrt((b_y1 * b_y1 + y2 * y2) + l_a * l_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b);
  m_a = y3 + 2.0 * a;
  n_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  o_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  p_a = y3 + 2.0 * a;
  q_a = y3 + 2.0 * a;
  x = sin(b);
  r_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  s_a = y3 + 2.0 * a;
  b_x = sin(b);
  t_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  u_a = y3 + 2.0 * a;
  u_a = (sqrt((b_y1 * b_y1 + y2 * y2) + u_a * u_a) + y3) + 2.0 * a;
  v_a = y3 + 2.0 * a;
  w_a = y3 + 2.0 * a;
  x_a = y3 + 2.0 * a;
  y_a = y3 + 2.0 * a;
  ab_a = y3 + 2.0 * a;
  ab_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ab_a * ab_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  bb_a = y3 + 2.0 * a;
  cb_a = y3 + 2.0 * a;
  db_a = y3 + 2.0 * a;
  eb_a = y3 + 2.0 * a;
  fb_a = y3 + 2.0 * a;
  gb_a = y3 + 2.0 * a;
  hb_a = y3 + 2.0 * a;
  ib_a = y3 + 2.0 * a;
  jb_a = y3 + 2.0 * a;
  kb_a = y3 + 2.0 * a;
  lb_a = y3 + 2.0 * a;
  mb_a = y3 + 2.0 * a;
  mb_a = (sqrt((b_y1 * b_y1 + y2 * y2) + mb_a * mb_a) + y3) + 2.0 * a;
  nb_a = y3 + 2.0 * a;
  ob_a = y3 + 2.0 * a;
  ob_a = (b_y1 * b_y1 + y2 * y2) + ob_a * ob_a;
  pb_a = y3 + 2.0 * a;
  qb_a = y3 + 2.0 * a;
  rb_a = y3 + 2.0 * a;
  sb_a = y3 + 2.0 * a;
  tb_a = y3 + 2.0 * a;
  ub_a = y3 + 2.0 * a;
  vb_a = y3 + 2.0 * a;
  wb_a = y3 + 2.0 * a;
  xb_a = y3 + 2.0 * a;
  yb_a = y3 + 2.0 * a;
  ac_a = y3 + 2.0 * a;
  bc_a = y3 + 2.0 * a;
  cc_a = y3 + 2.0 * a;
  dc_a = y3 + 2.0 * a;
  dc_a = (sqrt((b_y1 * b_y1 + y2 * y2) + dc_a * dc_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  ec_a = y3 + 2.0 * a;
  fc_a = y3 + 2.0 * a;
  gc_a = y3 + 2.0 * a;
  hc_a = y3 + 2.0 * a;
  ic_a = y3 + 2.0 * a;
  jc_a = y3 + 2.0 * a;
  kc_a = y3 + 2.0 * a;
  lc_a = y3 + 2.0 * a;
  mc_a = y3 + 2.0 * a;
  nc_a = y3 + 2.0 * a;
  oc_a = y3 + 2.0 * a;
  pc_a = y3 + 2.0 * a;
  pc_a = (sqrt((b_y1 * b_y1 + y2 * y2) + pc_a * pc_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  qc_a = y3 + 2.0 * a;
  rc_a = y3 + 2.0 * a;
  sc_a = y3 + 2.0 * a;
  tc_a = y3 + 2.0 * a;
  uc_a = y3 + 2.0 * a;
  vc_a = y3 + 2.0 * a;
  wc_a = y3 + 2.0 * a;
  wc_a = (b_y1 * b_y1 + y2 * y2) + wc_a * wc_a;
  xc_a = y3 + 2.0 * a;
  yc_a = y3 + 2.0 * a;
  ad_a = y3 + 2.0 * a;
  bd_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  cd_a = y3 + 2.0 * a;
  dd_a = y3 + 2.0 * a;
  ed_a = y3 + 2.0 * a;
  fd_a = y3 + 2.0 * a;
  gd_a = y3 + 2.0 * a;
  hd_a = y3 + 2.0 * a;
  id_a = y3 + 2.0 * a;
  jd_a = y3 + 2.0 * a;
  kd_a = y3 + 2.0 * a;
  ld_a = y3 + 2.0 * a;
  md_a = y3 + 2.0 * a;
  nd_a = y3 + 2.0 * a;
  nd_a = (sqrt((b_y1 * b_y1 + y2 * y2) + nd_a * nd_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  od_a = y3 + 2.0 * a;
  pd_a = y3 + 2.0 * a;
  qd_a = y3 + 2.0 * a;
  rd_a = y3 + 2.0 * a;
  sd_a = y3 + 2.0 * a;
  td_a = y3 + 2.0 * a;
  td_a = (sqrt((b_y1 * b_y1 + y2 * y2) + td_a * td_a) + y3) + 2.0 * a;
  ud_a = y3 + 2.0 * a;
  vd_a = y3 + 2.0 * a;
  wd_a = y3 + 2.0 * a;
  xd_a = y3 + 2.0 * a;
  yd_a = y3 + 2.0 * a;
  ae_a = y3 + 2.0 * a;
  be_a = y3 + 2.0 * a;
  be_a = (sqrt((b_y1 * b_y1 + y2 * y2) + be_a * be_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  ce_a = y3 + 2.0 * a;
  de_a = y3 + 2.0 * a;
  ee_a = y3 + 2.0 * a;
  fe_a = y3 + 2.0 * a;
  ge_a = y3 + 2.0 * a;
  he_a = y3 + 2.0 * a;
  ie_a = y3 + 2.0 * a;
  je_a = y3 + 2.0 * a;
  ke_a = y3 + 2.0 * a;
  le_a = y3 + 2.0 * a;
  me_a = y3 + 2.0 * a;
  ne_a = y3 + 2.0 * a;
  ne_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ne_a * ne_a) + y3) + 2.0 * a;
  oe_a = y3 + 2.0 * a;
  pe_a = y3 + 2.0 * a;
  pe_a = (b_y1 * b_y1 + y2 * y2) + pe_a * pe_a;
  qe_a = y3 + 2.0 * a;
  re_a = y3 + 2.0 * a;
  se_a = y3 + 2.0 * a;
  te_a = y3 + 2.0 * a;
  ue_a = y3 + 2.0 * a;
  ve_a = y3 + 2.0 * a;
  we_a = y3 + 2.0 * a;
  xe_a = y3 + 2.0 * a;
  ye_a = y3 + 2.0 * a;
  af_a = y3 + 2.0 * a;
  bf_a = y3 + 2.0 * a;
  bf_a = (sqrt((b_y1 * b_y1 + y2 * y2) + bf_a * bf_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  cf_a = y3 + 2.0 * a;
  df_a = y3 + 2.0 * a;
  ef_a = y3 + 2.0 * a;
  ff_a = y3 + 2.0 * a;
  gf_a = y3 + 2.0 * a;
  hf_a = y3 + 2.0 * a;
  if_a = y3 + 2.0 * a;
  jf_a = y3 + 2.0 * a;
  kf_a = y3 + 2.0 * a;
  lf_a = y3 + 2.0 * a;
  mf_a = y3 + 2.0 * a;
  nf_a = y3 + 2.0 * a;
  of_a = y3 + 2.0 * a;
  pf_a = y3 + 2.0 * a;
  qf_a = y3 + 2.0 * a;
  rf_a = y3 + 2.0 * a;
  sf_a = y3 + 2.0 * a;
  tf_a = y3 + 2.0 * a;
  uf_a = y3 + 2.0 * a;
  vf_a = y3 + 2.0 * a;
  wf_a = y3 + 2.0 * a;
  xf_a = y3 + 2.0 * a;
  yf_a = y3 + 2.0 * a;
  ag_a = y3 + 2.0 * a;
  bg_a = y3 + 2.0 * a;
  bg_a = (sqrt((b_y1 * b_y1 + y2 * y2) + bg_a * bg_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  cg_a = y3 + 2.0 * a;
  dg_a = y3 + 2.0 * a;
  eg_a = y3 + 2.0 * a;
  fg_a = y3 + 2.0 * a;
  gg_a = y3 + 2.0 * a;
  hg_a = y3 + 2.0 * a;
  ig_a = y3 + 2.0 * a;
  jg_a = y3 + 2.0 * a;
  kg_a = y3 + 2.0 * a;
  lg_a = y3 + 2.0 * a;
  lg_a = (b_y1 * b_y1 + y2 * y2) + lg_a * lg_a;
  mg_a = y3 + 2.0 * a;
  ng_a = y3 + 2.0 * a;
  og_a = y3 + 2.0 * a;
  pg_a = y3 + 2.0 * a;
  qg_a = y3 + 2.0 * a;
  rg_a = y3 + 2.0 * a;
  sg_a = y3 + 2.0 * a;
  tg_a = y3 + 2.0 * a;
  ug_a = y3 + 2.0 * a;
  vg_a = y3 + 2.0 * a;
  wg_a = y3 + 2.0 * a;
  xg_a = y3 + 2.0 * a;
  xg_a = (sqrt((b_y1 * b_y1 + y2 * y2) + xg_a * xg_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  yg_a = y3 + 2.0 * a;
  ah_a = b_y1 * cos(b) - y3 * sin(b);
  bh_a = b_y1 * cos(b) - y3 * sin(b);
  c_x = sin(b);
  ch_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  d_x = sin(b);
  dh_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  eh_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  fh_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  gh_a = y3 + 2.0 * a;
  hh_a = y3 + 2.0 * a;
  e_x = sin(b);
  ih_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  jh_a = y3 + 2.0 * a;
  f_x = sin(b);
  kh_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  lh_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  mh_a = y3 + 2.0 * a;
  nh_a = y3 + 2.0 * a;
  oh_a = y3 + 2.0 * a;
  ph_a = y3 + 2.0 * a;
  qh_a = y3 + 2.0 * a;
  rh_a = y3 + 2.0 * a;
  sh_a = y3 + 2.0 * a;
  th_a = y3 + 2.0 * a;
  uh_a = y3 + 2.0 * a;
  uh_a = (sqrt((b_y1 * b_y1 + y2 * y2) + uh_a * uh_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  vh_a = y3 + 2.0 * a;
  wh_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  xh_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  yh_a = y3 + 2.0 * a;
  ai_a = y3 + 2.0 * a;
  g_x = sin(b);
  bi_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  ci_a = y3 + 2.0 * a;
  h_x = sin(b);
  di_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  ei_a = y3 + 2.0 * a;
  ei_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ei_a * ei_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  fi_a = y3 + 2.0 * a;
  gi_a = y3 + 2.0 * a;
  hi_a = y3 + 2.0 * a;
  ii_a = y3 + 2.0 * a;
  ji_a = y3 + 2.0 * a;
  ki_a = y3 + 2.0 * a;
  li_a = y3 + 2.0 * a;
  mi_a = y3 + 2.0 * a;
  ni_a = y3 + 2.0 * a;
  oi_a = y3 + 2.0 * a;
  pi_a = y3 + 2.0 * a;
  qi_a = y3 + 2.0 * a;
  ri_a = y3 + 2.0 * a;
  si_a = y3 + 2.0 * a;
  ti_a = y3 + 2.0 * a;
  ui_a = y3 + 2.0 * a;
  vi_a = y3 + 2.0 * a;
  wi_a = y3 + 2.0 * a;
  wi_a = (sqrt((b_y1 * b_y1 + y2 * y2) + wi_a * wi_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  xi_a = y3 + 2.0 * a;
  yi_a = y3 + 2.0 * a;
  aj_a = y3 + 2.0 * a;
  bj_a = y3 + 2.0 * a;
  cj_a = y3 + 2.0 * a;
  dj_a = y3 + 2.0 * a;
  ej_a = y3 + 2.0 * a;
  fj_a = y3 + 2.0 * a;
  gj_a = y3 + 2.0 * a;
  hj_a = y3 + 2.0 * a;
  ij_a = y3 + 2.0 * a;
  jj_a = y3 + 2.0 * a;
  jj_a = (sqrt((b_y1 * b_y1 + y2 * y2) + jj_a * jj_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  kj_a = y3 + 2.0 * a;
  lj_a = y3 + 2.0 * a;
  mj_a = y3 + 2.0 * a;
  nj_a = y3 + 2.0 * a;
  oj_a = y3 + 2.0 * a;
  pj_a = y3 + 2.0 * a;
  qj_a = y3 + 2.0 * a;
  qj_a = (b_y1 * b_y1 + y2 * y2) + qj_a * qj_a;
  *e33 = (B1 * (0.125 * y2 * ((-1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 *
              y3, 1.5) * y3 + 0.5 / powd_snf((b_y1 * b_y1 + y2 * y2) + b_a *
              b_a, 1.5) * (2.0 * y3 + 4.0 * a)) - cos(b) * ((((((1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b) * y3 - 1.0) / sqrt((b_y1 * b_y1
    + y2 * y2) + y3 * y3) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 *
    sin(b)) - y3 * cos(b)) - (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b) -
    y3) / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) * y3) - (sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b) - y3) / sqrt((b_y1 * b_y1 + y2 *
    y2) + y3 * y3) / (c_a * c_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3)
    * y3 - cos(b))) - (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + d_a * d_a) * cos(b) *
                       (2.0 * y3 + 4.0 * a) + 1.0) / sqrt((b_y1 * b_y1 + y2 * y2)
    + e_a * e_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + f_a * f_a) - b_y1 * sin(b))
                    + (y3 + 2.0 * a) * cos(b))) + 0.5 * ((sqrt((b_y1 * b_y1 + y2
    * y2) + g_a * g_a) * cos(b) + y3) + 2.0 * a) / powd_snf((b_y1 * b_y1 + y2
    * y2) + h_a * h_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + i_a * i_a) - b_y1
    * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (2.0 * y3 + 4.0 * a)) + ((sqrt((b_y1 *
    b_y1 + y2 * y2) + j_a * j_a) * cos(b) + y3) + 2.0 * a) / sqrt((b_y1 * b_y1 +
    y2 * y2) + k_a * k_a) / (l_a * l_a) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) +
    m_a * m_a) * (2.0 * y3 + 4.0 * a) + cos(b)))) / M_PI / (1.0 -
            nu) + 0.25 * ((((((((2.0 - 2.0 * nu) * (((((1.0 - 2.0 * nu) * (-y2 /
    (n_a * n_a) * sin(b) / (1.0 + y2 * y2 / (o_a * o_a)) + (0.5 * y2 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + p_a * p_a) * sin(b) / (b_y1 * (b_y1 * cos(b) +
    (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b)) * (2.0 * y3 + 4.0 * a) - y2 *
    sqrt((b_y1 * b_y1 + y2 * y2) + q_a * q_a) * (x * x) / (r_a * r_a) * b_y1) /
    (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2) + s_a * s_a) * (b_x * b_x) / (t_a *
    t_a))) * (1.0 / tan(b)) - y2 / (u_a * u_a) * (2.0 * nu + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + v_a * v_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + w_a *
    w_a) * (2.0 * y3 + 4.0 * a) + 1.0)) - 0.5 * y2 / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + x_a * x_a) + y3) + 2.0 * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2)
    + y_a * y_a, 1.5) * (2.0 * y3 + 4.0 * a)) + y2 * cos(b) / (ab_a * ab_a) *
    (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + bb_a * bb_a)) * (0.5 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + cb_a * cb_a) * (2.0 * y3 + 4.0 * a) + cos(b))) +
    0.5 * y2 * cos(b) / ((sqrt((b_y1 * b_y1 + y2 * y2) + db_a * db_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 * y2)
    + eb_a * eb_a, 1.5) * (2.0 * y3 + 4.0 * a)) + y2 / sqrt((b_y1 * b_y1 + y2 *
    y2) + fb_a * fb_a) * (2.0 * nu / ((sqrt((b_y1 * b_y1 + y2 * y2) + gb_a *
    gb_a) + y3) + 2.0 * a) + a / ((b_y1 * b_y1 + y2 * y2) + hb_a * hb_a))) - 0.5
    * y2 * (y3 + a) / powd_snf((b_y1 * b_y1 + y2 * y2) + ib_a * ib_a, 1.5) *
    (2.0 * nu / ((sqrt((b_y1 * b_y1 + y2 * y2) + jb_a * jb_a) + y3) + 2.0 * a) +
     a / ((b_y1 * b_y1 + y2 * y2) + kb_a * kb_a)) * (2.0 * y3 + 4.0 * a)) + y2 *
    (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) + lb_a * lb_a) * (-2.0 * nu / (mb_a *
    mb_a) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + nb_a * nb_a) * (2.0 * y3 + 4.0
    * a) + 1.0) - a / (ob_a * ob_a) * (2.0 * y3 + 4.0 * a))) + y2 * cos(b) /
    sqrt((b_y1 * b_y1 + y2 * y2) + pb_a * pb_a) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + qb_a * qb_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((1.0 - 2.0 *
    nu) - ((sqrt((b_y1 * b_y1 + y2 * y2) + rb_a * rb_a) * cos(b) + y3) + 2.0 * a)
    / ((sqrt((b_y1 * b_y1 + y2 * y2) + sb_a * sb_a) - b_y1 * sin(b)) + (y3 + 2.0
    * a) * cos(b)) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + tb_a * tb_a)))
    - a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) + ub_a * ub_a))) - 0.5 * y2 *
              (y3 + a) * cos(b) / powd_snf((b_y1 * b_y1 + y2 * y2) + vb_a *
    vb_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + wb_a * wb_a) - b_y1 * sin(b))
                  + (y3 + 2.0 * a) * cos(b)) * (((1.0 - 2.0 * nu) - ((sqrt((b_y1
    * b_y1 + y2 * y2) + xb_a * xb_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + yb_a * yb_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + ac_a * ac_a))) - a * (y3 + 2.0 *
    a) / ((b_y1 * b_y1 + y2 * y2) + bc_a * bc_a)) * (2.0 * y3 + 4.0 * a)) - y2 *
             (y3 + a) * cos(b) / sqrt((b_y1 * b_y1 + y2 * y2) + cc_a * cc_a) /
             (dc_a * dc_a) * (((1.0 - 2.0 * nu) - ((sqrt((b_y1 * b_y1 + y2 * y2)
    + ec_a * ec_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    fc_a * fc_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + gc_a * gc_a))) - a * (y3 + 2.0 * a) / ((b_y1 *
    b_y1 + y2 * y2) + hc_a * hc_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + ic_a
    * ic_a) * (2.0 * y3 + 4.0 * a) + cos(b))) + y2 * (y3 + a) * cos(b) / sqrt
            ((b_y1 * b_y1 + y2 * y2) + jc_a * jc_a) / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + kc_a * kc_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((((-(0.5 /
    sqrt((b_y1 * b_y1 + y2 * y2) + lc_a * lc_a) * cos(b) * (2.0 * y3 + 4.0 * a)
    + 1.0) / ((sqrt((b_y1 * b_y1 + y2 * y2) + mc_a * mc_a) - b_y1 * sin(b)) +
              (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 *
    y2) + nc_a * nc_a)) + ((sqrt((b_y1 * b_y1 + y2 * y2) + oc_a * oc_a) * cos(b)
    + y3) + 2.0 * a) / (pc_a * pc_a) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2)
    + qc_a * qc_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + rc_a * rc_a) * (2.0 *
    y3 + 4.0 * a) + cos(b))) + 0.5 * ((sqrt((b_y1 * b_y1 + y2 * y2) + sc_a *
    sc_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + tc_a *
    tc_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 *
    b_y1 + y2 * y2) + uc_a * uc_a, 1.5) * (2.0 * y3 + 4.0 * a)) - a / ((b_y1 *
    b_y1 + y2 * y2) + vc_a * vc_a)) + a * (y3 + 2.0 * a) / (wc_a * wc_a) * (2.0 *
              y3 + 4.0 * a))) / M_PI / (1.0 - nu)) + B2 * (0.125 *
           ((((((((((-1.0 + 2.0 * nu) * sin(b) * ((1.0 / sqrt((b_y1 * b_y1 + y2 *
    y2) + y3 * y3) * y3 - cos(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) -
    b_y1 * sin(b)) - y3 * cos(b)) - (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + xc_a *
    xc_a) * (2.0 * y3 + 4.0 * a) + cos(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    yc_a * yc_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) - b_y1 * (-1.0 /
    powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) * y3 + 0.5 / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + ad_a * ad_a, 1.5) * (2.0 * y3 + 4.0 * a))) - sin
                   (b) * (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b) - y3)
                   / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b))) + (b_y1 * cos(b)
    - y3 * sin(b)) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b) *
                      y3 - 1.0) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) /
                  ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) -
                   y3 * cos(b))) - (b_y1 * cos(b) - y3 * sin(b)) * (sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) * cos(b) - y3) / powd_snf((b_y1 * b_y1 + y2 *
    y2) + y3 * y3, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin
    (b)) - y3 * cos(b)) * y3) - (b_y1 * cos(b) - y3 * sin(b)) * (sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) * cos(b) - y3) / sqrt((b_y1 * b_y1 + y2 * y2) +
    y3 * y3) / (bd_a * bd_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) *
    y3 - cos(b))) - sin(b) * ((sqrt((b_y1 * b_y1 + y2 * y2) + cd_a * cd_a) * cos
    (b) + y3) + 2.0 * a) / sqrt((b_y1 * b_y1 + y2 * y2) + dd_a * dd_a) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + ed_a * ed_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b))) - (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (0.5 / sqrt((b_y1 *
    b_y1 + y2 * y2) + fd_a * fd_a) * cos(b) * (2.0 * y3 + 4.0 * a) + 1.0) / sqrt
              ((b_y1 * b_y1 + y2 * y2) + gd_a * gd_a) / ((sqrt((b_y1 * b_y1 + y2
    * y2) + hd_a * hd_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) + 0.5 *
             (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * ((sqrt((b_y1 * b_y1 +
    y2 * y2) + id_a * id_a) * cos(b) + y3) + 2.0 * a) / powd_snf((b_y1 * b_y1
    + y2 * y2) + jd_a * jd_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + kd_a *
    kd_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (2.0 * y3 + 4.0 * a)) +
            (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * ((sqrt((b_y1 * b_y1 + y2
    * y2) + ld_a * ld_a) * cos(b) + y3) + 2.0 * a) / sqrt((b_y1 * b_y1 + y2 * y2)
             + md_a * md_a) / (nd_a * nd_a) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2)
              + od_a * od_a) * (2.0 * y3 + 4.0 * a) + cos(b))) /
           M_PI / (1.0 - nu) + 0.25 * ((((((((((((-2.0 + 2.0 * nu)
    * (1.0 - 2.0 * nu) * (1.0 / tan(b)) * ((0.5 / sqrt((b_y1 * b_y1 + y2 * y2) +
    pd_a * pd_a) * (2.0 * y3 + 4.0 * a) + 1.0) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + qd_a * qd_a) + y3) + 2.0 * a) - cos(b) * (0.5 / sqrt((b_y1 * b_y1 + y2 *
    y2) + rd_a * rd_a) * (2.0 * y3 + 4.0 * a) + cos(b)) / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + sd_a * sd_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) + (2.0
    - 2.0 * nu) * b_y1 / (td_a * td_a) * (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 *
    y2) + ud_a * ud_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + vd_a * vd_a) *
    (2.0 * y3 + 4.0 * a) + 1.0)) + 0.5 * (2.0 - 2.0 * nu) * b_y1 / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + wd_a * wd_a) + y3) + 2.0 * a) * a / powd_snf((b_y1 *
    b_y1 + y2 * y2) + xd_a * xd_a, 1.5) * (2.0 * y3 + 4.0 * a)) + (2.0 - 2.0 *
    nu) * sin(b) / ((sqrt((b_y1 * b_y1 + y2 * y2) + yd_a * yd_a) - b_y1 * sin(b))
                    + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1 * b_y1
    + y2 * y2) + ae_a * ae_a))) - (2.0 - 2.0 * nu) * (b_y1 * cos(b) + (y3 + 2.0 *
    a) * sin(b)) / (be_a * be_a) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) +
    ce_a * ce_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + de_a * de_a) * (2.0 *
    y3 + 4.0 * a) + cos(b))) - 0.5 * (2.0 - 2.0 * nu) * (b_y1 * cos(b) + (y3 +
    2.0 * a) * sin(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ee_a * ee_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 * y2)
    + fe_a * fe_a, 1.5) * (2.0 * y3 + 4.0 * a)) + 1.0 / sqrt((b_y1 * b_y1 + y2 *
    y2) + ge_a * ge_a) * (((1.0 - 2.0 * nu) * (1.0 / tan(b)) - 2.0 * nu * b_y1 /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + he_a * he_a) + y3) + 2.0 * a)) - a * b_y1 /
    ((b_y1 * b_y1 + y2 * y2) + ie_a * ie_a))) - 0.5 * (y3 + a) / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + je_a * je_a, 1.5) * (((1.0 - 2.0 * nu) * (1.0 /
    tan(b)) - 2.0 * nu * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + ke_a * ke_a) +
    y3) + 2.0 * a)) - a * b_y1 / ((b_y1 * b_y1 + y2 * y2) + le_a * le_a)) * (2.0
    * y3 + 4.0 * a)) + (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) + me_a * me_a) *
    (2.0 * nu * b_y1 / (ne_a * ne_a) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) +
    oe_a * oe_a) * (2.0 * y3 + 4.0 * a) + 1.0) + a * b_y1 / (pe_a * pe_a) * (2.0
    * y3 + 4.0 * a))) - 1.0 / ((sqrt((b_y1 * b_y1 + y2 * y2) + qe_a * qe_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((cos(b) * sin(b) + ((sqrt((b_y1
    * b_y1 + y2 * y2) + re_a * re_a) * cos(b) + y3) + 2.0 * a) * (1.0 / tan(b)) /
    sqrt((b_y1 * b_y1 + y2 * y2) + se_a * se_a) * ((2.0 - 2.0 * nu) * cos(b) -
    ((sqrt((b_y1 * b_y1 + y2 * y2) + te_a * te_a) * cos(b) + y3) + 2.0 * a) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + ue_a * ue_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)))) + a / sqrt((b_y1 * b_y1 + y2 * y2) + ve_a * ve_a) * ((sin(b)
    - (y3 + 2.0 * a) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / ((b_y1 * b_y1
    + y2 * y2) + we_a * we_a)) - (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) *
    ((sqrt((b_y1 * b_y1 + y2 * y2) + xe_a * xe_a) * cos(b) + y3) + 2.0 * a) /
    sqrt((b_y1 * b_y1 + y2 * y2) + ye_a * ye_a) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + af_a * af_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))))) + (y3 + a) /
             (bf_a * bf_a) * ((cos(b) * sin(b) + ((sqrt((b_y1 * b_y1 + y2 * y2)
    + cf_a * cf_a) * cos(b) + y3) + 2.0 * a) * (1.0 / tan(b)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + df_a * df_a) * ((2.0 - 2.0 * nu) * cos(b) - ((sqrt((b_y1 *
    b_y1 + y2 * y2) + ef_a * ef_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + ff_a * ff_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))))
              + a / sqrt((b_y1 * b_y1 + y2 * y2) + gf_a * gf_a) * ((sin(b) - (y3
    + 2.0 * a) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / ((b_y1 * b_y1 + y2 *
    y2) + hf_a * hf_a)) - (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + if_a * if_a) * cos(b) + y3) + 2.0 * a) / sqrt
    ((b_y1 * b_y1 + y2 * y2) + jf_a * jf_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    kf_a * kf_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)))) * (0.5 / sqrt
              ((b_y1 * b_y1 + y2 * y2) + lf_a * lf_a) * (2.0 * y3 + 4.0 * a) +
              cos(b))) - (y3 + a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + mf_a * mf_a)
              - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((((0.5 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + nf_a * nf_a) * cos(b) * (2.0 * y3 + 4.0 * a) +
    1.0) * (1.0 / tan(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + of_a * of_a) * ((2.0
    - 2.0 * nu) * cos(b) - ((sqrt((b_y1 * b_y1 + y2 * y2) + pf_a * pf_a) * cos(b)
    + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + qf_a * qf_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b))) - 0.5 * ((sqrt((b_y1 * b_y1 + y2 * y2) +
    rf_a * rf_a) * cos(b) + y3) + 2.0 * a) * (1.0 / tan(b)) / powd_snf((b_y1 *
    b_y1 + y2 * y2) + sf_a * sf_a, 1.5) * ((2.0 - 2.0 * nu) * cos(b) - ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + tf_a * tf_a) * cos(b) + y3) + 2.0 * a) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + uf_a * uf_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b))) * (2.0 * y3 + 4.0 * a)) + ((sqrt((b_y1 * b_y1 + y2 * y2) + vf_a *
    vf_a) * cos(b) + y3) + 2.0 * a) * (1.0 / tan(b)) / sqrt((b_y1 * b_y1 + y2 *
    y2) + wf_a * wf_a) * (-(0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + xf_a * xf_a) *
    cos(b) * (2.0 * y3 + 4.0 * a) + 1.0) / ((sqrt((b_y1 * b_y1 + y2 * y2) + yf_a
    * yf_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) + ((sqrt((b_y1 * b_y1 +
    y2 * y2) + ag_a * ag_a) * cos(b) + y3) + 2.0 * a) / (bg_a * bg_a) * (0.5 /
    sqrt((b_y1 * b_y1 + y2 * y2) + cg_a * cg_a) * (2.0 * y3 + 4.0 * a) + cos(b))))
              - 0.5 * a / powd_snf((b_y1 * b_y1 + y2 * y2) + dg_a * dg_a, 1.5)
              * ((sin(b) - (y3 + 2.0 * a) * (b_y1 * cos(b) + (y3 + 2.0 * a) *
    sin(b)) / ((b_y1 * b_y1 + y2 * y2) + eg_a * eg_a)) - (b_y1 * cos(b) + (y3 +
    2.0 * a) * sin(b)) * ((sqrt((b_y1 * b_y1 + y2 * y2) + fg_a * fg_a) * cos(b)
    + y3) + 2.0 * a) / sqrt((b_y1 * b_y1 + y2 * y2) + gg_a * gg_a) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + hg_a * hg_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b))) * (2.0 * y3 + 4.0 * a)) + a / sqrt((b_y1 * b_y1 + y2 * y2) + ig_a *
              ig_a) * ((((((-(b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / ((b_y1 *
    b_y1 + y2 * y2) + jg_a * jg_a) - (y3 + 2.0 * a) * sin(b) / ((b_y1 * b_y1 +
    y2 * y2) + kg_a * kg_a)) + (y3 + 2.0 * a) * (b_y1 * cos(b) + (y3 + 2.0 * a) *
    sin(b)) / (lg_a * lg_a) * (2.0 * y3 + 4.0 * a)) - sin(b) * ((sqrt((b_y1 *
    b_y1 + y2 * y2) + mg_a * mg_a) * cos(b) + y3) + 2.0 * a) / sqrt((b_y1 * b_y1
    + y2 * y2) + ng_a * ng_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + og_a * og_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) - (b_y1 * cos(b) + (y3 + 2.0 * a)
    * sin(b)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + pg_a * pg_a) * cos(b) *
                 (2.0 * y3 + 4.0 * a) + 1.0) / sqrt((b_y1 * b_y1 + y2 * y2) +
    qg_a * qg_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + rg_a * rg_a) - b_y1 * sin(b))
                    + (y3 + 2.0 * a) * cos(b))) + 0.5 * (b_y1 * cos(b) + (y3 +
    2.0 * a) * sin(b)) * ((sqrt((b_y1 * b_y1 + y2 * y2) + sg_a * sg_a) * cos(b)
    + y3) + 2.0 * a) / powd_snf((b_y1 * b_y1 + y2 * y2) + tg_a * tg_a, 1.5) /
                        ((sqrt((b_y1 * b_y1 + y2 * y2) + ug_a * ug_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * (2.0 * y3 + 4.0 * a)) + (b_y1 * cos(b)
    + (y3 + 2.0 * a) * sin(b)) * ((sqrt((b_y1 * b_y1 + y2 * y2) + vg_a * vg_a) *
    cos(b) + y3) + 2.0 * a) / sqrt((b_y1 * b_y1 + y2 * y2) + wg_a * wg_a) /
                       (xg_a * xg_a) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) +
    yg_a * yg_a) * (2.0 * y3 + 4.0 * a) + cos(b))))) / M_PI / (1.0
            - nu))) + B3 * (0.125 * ((2.0 - 2.0 * nu) * (((y2 / (ah_a * ah_a) *
    sin(b) / (1.0 + y2 * y2 / (bh_a * bh_a)) + (y2 / sqrt((b_y1 * b_y1 + y2 * y2)
    + y3 * y3) * sin(b) / (b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos
    (b)) * y3 + y2 * sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * (c_x * c_x) /
    (ch_a * ch_a) * b_y1) / (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2) + y3 * y3)
    * (d_x * d_x) / (dh_a * dh_a))) + y2 / (eh_a * eh_a) * sin(b) / (1.0 + y2 *
    y2 / (fh_a * fh_a))) - (0.5 * y2 / sqrt((b_y1 * b_y1 + y2 * y2) + gh_a *
    gh_a) * sin(b) / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2
                      * cos(b)) * (2.0 * y3 + 4.0 * a) - y2 * sqrt((b_y1 * b_y1
    + y2 * y2) + hh_a * hh_a) * (e_x * e_x) / (ih_a * ih_a) * b_y1) / (1.0 + y2 *
    y2 * ((b_y1 * b_y1 + y2 * y2) + jh_a * jh_a) * (f_x * f_x) / (kh_a * kh_a)))
    + y2 * sin(b) * ((((((1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b)
    * y3 - 1.0) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / ((sqrt((b_y1 * b_y1
    + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) - (sqrt((b_y1 * b_y1 +
    y2 * y2) + y3 * y3) * cos(b) - y3) / powd_snf((b_y1 * b_y1 + y2 * y2) +
    y3 * y3, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) -
                     y3 * cos(b)) * y3) - (sqrt((b_y1 * b_y1 + y2 * y2) + y3 *
    y3) * cos(b) - y3) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (lh_a * lh_a)
                        * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * y3 -
    cos(b))) - (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + mh_a * mh_a) * cos(b) *
                (2.0 * y3 + 4.0 * a) + 1.0) / sqrt((b_y1 * b_y1 + y2 * y2) +
    nh_a * nh_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + oh_a * oh_a) - b_y1 * sin(b))
                    + (y3 + 2.0 * a) * cos(b))) + 0.5 * ((sqrt((b_y1 * b_y1 + y2
    * y2) + ph_a * ph_a) * cos(b) + y3) + 2.0 * a) / powd_snf((b_y1 * b_y1 +
    y2 * y2) + qh_a * qh_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + rh_a * rh_a)
    - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (2.0 * y3 + 4.0 * a)) + ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + sh_a * sh_a) * cos(b) + y3) + 2.0 * a) / sqrt
                     ((b_y1 * b_y1 + y2 * y2) + th_a * th_a) / (uh_a * uh_a) *
                     (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + vh_a * vh_a) * (2.0 *
    y3 + 4.0 * a) + cos(b)))) / M_PI / (1.0 - nu) + 0.25 *
    (((((((2.0 - 2.0 * nu) * (-y2 / (wh_a * wh_a) * sin(b) / (1.0 + y2 * y2 /
    (xh_a * xh_a)) + (0.5 * y2 / sqrt((b_y1 * b_y1 + y2 * y2) + yh_a * yh_a) *
                      sin(b) / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b))
    + y2 * y2 * cos(b)) * (2.0 * y3 + 4.0 * a) - y2 * sqrt((b_y1 * b_y1 + y2 *
    y2) + ai_a * ai_a) * (g_x * g_x) / (bi_a * bi_a) * b_y1) / (1.0 + y2 * y2 *
    ((b_y1 * b_y1 + y2 * y2) + ci_a * ci_a) * (h_x * h_x) / (di_a * di_a))) -
          (2.0 - 2.0 * nu) * y2 * sin(b) / (ei_a * ei_a) * (cos(b) + a / sqrt
    ((b_y1 * b_y1 + y2 * y2) + fi_a * fi_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 *
    y2) + gi_a * gi_a) * (2.0 * y3 + 4.0 * a) + cos(b))) - 0.5 * (2.0 - 2.0 * nu)
         * y2 * sin(b) / ((sqrt((b_y1 * b_y1 + y2 * y2) + hi_a * hi_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 * y2)
    + ii_a * ii_a, 1.5) * (2.0 * y3 + 4.0 * a)) + y2 * sin(b) / sqrt((b_y1 *
    b_y1 + y2 * y2) + ji_a * ji_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ki_a *
    ki_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((1.0 + ((sqrt((b_y1 *
    b_y1 + y2 * y2) + li_a * li_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + mi_a * mi_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + ni_a * ni_a))) + a * (y3 + 2.0 *
    a) / ((b_y1 * b_y1 + y2 * y2) + oi_a * oi_a))) - 0.5 * y2 * (y3 + a) * sin(b)
       / powd_snf((b_y1 * b_y1 + y2 * y2) + pi_a * pi_a, 1.5) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + qi_a * qi_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
       ((1.0 + ((sqrt((b_y1 * b_y1 + y2 * y2) + ri_a * ri_a) * cos(b) + y3) +
                2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + si_a * si_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 *
    y2) + ti_a * ti_a))) + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) + ui_a *
    ui_a)) * (2.0 * y3 + 4.0 * a)) - y2 * (y3 + a) * sin(b) / sqrt((b_y1 * b_y1
    + y2 * y2) + vi_a * vi_a) / (wi_a * wi_a) * ((1.0 + ((sqrt((b_y1 * b_y1 + y2
    * y2) + xi_a * xi_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + yi_a * yi_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a
    / sqrt((b_y1 * b_y1 + y2 * y2) + aj_a * aj_a))) + a * (y3 + 2.0 * a) /
    ((b_y1 * b_y1 + y2 * y2) + bj_a * bj_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 *
    y2) + cj_a * cj_a) * (2.0 * y3 + 4.0 * a) + cos(b))) + y2 * (y3 + a) * sin(b)
     / sqrt((b_y1 * b_y1 + y2 * y2) + dj_a * dj_a) / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + ej_a * ej_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((((0.5 /
    sqrt((b_y1 * b_y1 + y2 * y2) + fj_a * fj_a) * cos(b) * (2.0 * y3 + 4.0 * a)
    + 1.0) / ((sqrt((b_y1 * b_y1 + y2 * y2) + gj_a * gj_a) - b_y1 * sin(b)) +
              (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 *
    y2) + hj_a * hj_a)) - ((sqrt((b_y1 * b_y1 + y2 * y2) + ij_a * ij_a) * cos(b)
    + y3) + 2.0 * a) / (jj_a * jj_a) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2)
    + kj_a * kj_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + lj_a * lj_a) * (2.0 *
    y3 + 4.0 * a) + cos(b))) - 0.5 * ((sqrt((b_y1 * b_y1 + y2 * y2) + mj_a *
    mj_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + nj_a *
    nj_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 *
    b_y1 + y2 * y2) + oj_a * oj_a, 1.5) * (2.0 * y3 + 4.0 * a)) + a / ((b_y1 *
    b_y1 + y2 * y2) + pj_a * pj_a)) - a * (y3 + 2.0 * a) / (qj_a * qj_a) * (2.0 *
    y3 + 4.0 * a))) / M_PI / (1.0 - nu));
  b_a = b_y1 * cos(b) - y3 * sin(b);
  c_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  x = sin(b);
  d_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  e_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  f_a = y3 + 2.0 * a;
  g_a = y3 + 2.0 * a;
  h_a = y3 + 2.0 * a;
  i_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  j_a = y3 + 2.0 * a;
  b_x = sin(b);
  k_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  l_a = y3 + 2.0 * a;
  m_a = y3 + 2.0 * a;
  n_a = sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3;
  o_a = y3 + 2.0 * a;
  p_a = y3 + 2.0 * a;
  q_a = y3 + 2.0 * a;
  r_a = y3 + 2.0 * a;
  r_a = (sqrt((b_y1 * b_y1 + y2 * y2) + r_a * r_a) + y3) + 2.0 * a;
  s_a = y3 + 2.0 * a;
  t_a = y3 + 2.0 * a;
  u_a = y3 + 2.0 * a;
  v_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  w_a = y3 + 2.0 * a;
  x_a = y3 + 2.0 * a;
  y_a = y3 + 2.0 * a;
  ab_a = y3 + 2.0 * a;
  bb_a = y3 + 2.0 * a;
  cb_a = y3 + 2.0 * a;
  db_a = y3 + 2.0 * a;
  eb_a = y3 + 2.0 * a;
  eb_a = (sqrt((b_y1 * b_y1 + y2 * y2) + eb_a * eb_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  fb_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  gb_a = y3 + 2.0 * a;
  hb_a = y3 + 2.0 * a;
  ib_a = y3 + 2.0 * a;
  jb_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  kb_a = y3 + 2.0 * a;
  c_x = sin(b);
  lb_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  d_x = 1.0 / tan(b);
  mb_a = y3 + 2.0 * a;
  nb_a = y3 + 2.0 * a;
  ob_a = y3 + 2.0 * a;
  pb_a = y3 + 2.0 * a;
  qb_a = y3 + 2.0 * a;
  qb_a = (sqrt((b_y1 * b_y1 + y2 * y2) + qb_a * qb_a) + y3) + 2.0 * a;
  rb_a = y3 + 2.0 * a;
  sb_a = y3 + 2.0 * a;
  tb_a = y3 + 2.0 * a;
  ub_a = y3 + 2.0 * a;
  vb_a = y3 + 2.0 * a;
  wb_a = y3 + 2.0 * a;
  xb_a = y3 + 2.0 * a;
  xb_a = (sqrt((b_y1 * b_y1 + y2 * y2) + xb_a * xb_a) + y3) + 2.0 * a;
  yb_a = y3 + 2.0 * a;
  ac_a = y3 + 2.0 * a;
  bc_a = y3 + 2.0 * a;
  cc_a = y3 + 2.0 * a;
  dc_a = y3 + 2.0 * a;
  ec_a = y3 + 2.0 * a;
  fc_a = y3 + 2.0 * a;
  fc_a = (sqrt((b_y1 * b_y1 + y2 * y2) + fc_a * fc_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  gc_a = y3 + 2.0 * a;
  hc_a = y3 + 2.0 * a;
  ic_a = y3 + 2.0 * a;
  jc_a = y3 + 2.0 * a;
  kc_a = y3 + 2.0 * a;
  lc_a = y3 + 2.0 * a;
  mc_a = y3 + 2.0 * a;
  nc_a = y3 + 2.0 * a;
  oc_a = y3 + 2.0 * a;
  pc_a = y3 + 2.0 * a;
  qc_a = y3 + 2.0 * a;
  rc_a = y3 + 2.0 * a;
  sc_a = y3 + 2.0 * a;
  tc_a = y3 + 2.0 * a;
  uc_a = y3 + 2.0 * a;
  vc_a = y3 + 2.0 * a;
  wc_a = y3 + 2.0 * a;
  xc_a = y3 + 2.0 * a;
  xc_a = (sqrt((b_y1 * b_y1 + y2 * y2) + xc_a * xc_a) + y3) + 2.0 * a;
  yc_a = y3 + 2.0 * a;
  ad_a = y3 + 2.0 * a;
  bd_a = y3 + 2.0 * a;
  cd_a = y3 + 2.0 * a;
  dd_a = y3 + 2.0 * a;
  ed_a = y3 + 2.0 * a;
  ed_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ed_a * ed_a) + y3) + 2.0 * a;
  fd_a = y3 + 2.0 * a;
  gd_a = y3 + 2.0 * a;
  hd_a = y3 + 2.0 * a;
  id_a = y3 + 2.0 * a;
  jd_a = y3 + 2.0 * a;
  jd_a = (b_y1 * b_y1 + y2 * y2) + jd_a * jd_a;
  kd_a = y3 + 2.0 * a;
  ld_a = y3 + 2.0 * a;
  md_a = y3 + 2.0 * a;
  nd_a = y3 + 2.0 * a;
  od_a = y3 + 2.0 * a;
  pd_a = y3 + 2.0 * a;
  qd_a = y3 + 2.0 * a;
  rd_a = y3 + 2.0 * a;
  sd_a = y3 + 2.0 * a;
  td_a = y3 + 2.0 * a;
  ud_a = y3 + 2.0 * a;
  vd_a = y3 + 2.0 * a;
  wd_a = y3 + 2.0 * a;
  xd_a = y3 + 2.0 * a;
  yd_a = y3 + 2.0 * a;
  ae_a = y3 + 2.0 * a;
  ae_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ae_a * ae_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  be_a = y3 + 2.0 * a;
  ce_a = y3 + 2.0 * a;
  de_a = y3 + 2.0 * a;
  ee_a = y3 + 2.0 * a;
  fe_a = y3 + 2.0 * a;
  ge_a = y3 + 2.0 * a;
  he_a = y3 + 2.0 * a;
  ie_a = y3 + 2.0 * a;
  ie_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ie_a * ie_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  je_a = y3 + 2.0 * a;
  ke_a = y3 + 2.0 * a;
  le_a = y3 + 2.0 * a;
  me_a = y3 + 2.0 * a;
  ne_a = y3 + 2.0 * a;
  oe_a = y3 + 2.0 * a;
  pe_a = y3 + 2.0 * a;
  qe_a = y3 + 2.0 * a;
  re_a = y3 + 2.0 * a;
  se_a = y3 + 2.0 * a;
  te_a = y3 + 2.0 * a;
  te_a = (b_y1 * b_y1 + y2 * y2) + te_a * te_a;
  ue_a = y3 + 2.0 * a;
  ve_a = y3 + 2.0 * a;
  we_a = y3 + 2.0 * a;
  xe_a = y3 + 2.0 * a;
  ye_a = sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3;
  af_a = y3 + 2.0 * a;
  bf_a = y3 + 2.0 * a;
  cf_a = y3 + 2.0 * a;
  df_a = y3 + 2.0 * a;
  df_a = (sqrt((b_y1 * b_y1 + y2 * y2) + df_a * df_a) + y3) + 2.0 * a;
  ef_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  ff_a = y3 + 2.0 * a;
  gf_a = y3 + 2.0 * a;
  hf_a = y3 + 2.0 * a;
  if_a = y3 + 2.0 * a;
  jf_a = y3 + 2.0 * a;
  kf_a = y3 + 2.0 * a;
  lf_a = y3 + 2.0 * a;
  mf_a = y3 + 2.0 * a;
  mf_a = (sqrt((b_y1 * b_y1 + y2 * y2) + mf_a * mf_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  e_x = 1.0 / tan(b);
  nf_a = y3 + 2.0 * a;
  of_a = y3 + 2.0 * a;
  f_x = 1.0 / tan(b);
  pf_a = y3 + 2.0 * a;
  qf_a = y3 + 2.0 * a;
  rf_a = y3 + 2.0 * a;
  rf_a = (sqrt((b_y1 * b_y1 + y2 * y2) + rf_a * rf_a) + y3) + 2.0 * a;
  sf_a = y3 + 2.0 * a;
  tf_a = y3 + 2.0 * a;
  uf_a = y3 + 2.0 * a;
  vf_a = y3 + 2.0 * a;
  wf_a = y3 + 2.0 * a;
  xf_a = y3 + 2.0 * a;
  yf_a = y3 + 2.0 * a;
  yf_a = (sqrt((b_y1 * b_y1 + y2 * y2) + yf_a * yf_a) + y3) + 2.0 * a;
  ag_a = y3 + 2.0 * a;
  bg_a = y3 + 2.0 * a;
  cg_a = y3 + 2.0 * a;
  dg_a = y3 + 2.0 * a;
  eg_a = y3 + 2.0 * a;
  eg_a = (sqrt((b_y1 * b_y1 + y2 * y2) + eg_a * eg_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  fg_a = y3 + 2.0 * a;
  gg_a = y3 + 2.0 * a;
  hg_a = y3 + 2.0 * a;
  ig_a = y3 + 2.0 * a;
  jg_a = y3 + 2.0 * a;
  kg_a = y3 + 2.0 * a;
  lg_a = y3 + 2.0 * a;
  mg_a = y3 + 2.0 * a;
  ng_a = y3 + 2.0 * a;
  ng_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ng_a * ng_a) + y3) + 2.0 * a;
  og_a = y3 + 2.0 * a;
  pg_a = y3 + 2.0 * a;
  qg_a = y3 + 2.0 * a;
  rg_a = y3 + 2.0 * a;
  sg_a = y3 + 2.0 * a;
  tg_a = y3 + 2.0 * a;
  ug_a = y3 + 2.0 * a;
  vg_a = y3 + 2.0 * a;
  wg_a = y3 + 2.0 * a;
  xg_a = y3 + 2.0 * a;
  yg_a = y3 + 2.0 * a;
  ah_a = y3 + 2.0 * a;
  bh_a = y3 + 2.0 * a;
  bh_a = (sqrt((b_y1 * b_y1 + y2 * y2) + bh_a * bh_a) + y3) + 2.0 * a;
  ch_a = y3 + 2.0 * a;
  dh_a = y3 + 2.0 * a;
  dh_a = (b_y1 * b_y1 + y2 * y2) + dh_a * dh_a;
  eh_a = y3 + 2.0 * a;
  fh_a = y3 + 2.0 * a;
  gh_a = y3 + 2.0 * a;
  gh_a = (sqrt((b_y1 * b_y1 + y2 * y2) + gh_a * gh_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  hh_a = y3 + 2.0 * a;
  ih_a = y3 + 2.0 * a;
  jh_a = y3 + 2.0 * a;
  kh_a = y3 + 2.0 * a;
  lh_a = y3 + 2.0 * a;
  mh_a = y3 + 2.0 * a;
  nh_a = y3 + 2.0 * a;
  oh_a = y3 + 2.0 * a;
  ph_a = y3 + 2.0 * a;
  qh_a = y3 + 2.0 * a;
  rh_a = y3 + 2.0 * a;
  sh_a = y3 + 2.0 * a;
  th_a = y3 + 2.0 * a;
  uh_a = y3 + 2.0 * a;
  vh_a = y3 + 2.0 * a;
  wh_a = y3 + 2.0 * a;
  xh_a = y3 + 2.0 * a;
  yh_a = y3 + 2.0 * a;
  ai_a = y3 + 2.0 * a;
  bi_a = y3 + 2.0 * a;
  ci_a = y3 + 2.0 * a;
  di_a = y3 + 2.0 * a;
  ei_a = y3 + 2.0 * a;
  fi_a = y3 + 2.0 * a;
  gi_a = y3 + 2.0 * a;
  gi_a = (sqrt((b_y1 * b_y1 + y2 * y2) + gi_a * gi_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  hi_a = y3 + 2.0 * a;
  ii_a = y3 + 2.0 * a;
  ji_a = y3 + 2.0 * a;
  ki_a = y3 + 2.0 * a;
  li_a = y3 + 2.0 * a;
  mi_a = y3 + 2.0 * a;
  ni_a = y3 + 2.0 * a;
  oi_a = y3 + 2.0 * a;
  pi_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  qi_a = y3 + 2.0 * a;
  ri_a = y3 + 2.0 * a;
  si_a = y3 + 2.0 * a;
  ti_a = y3 + 2.0 * a;
  ui_a = y3 + 2.0 * a;
  vi_a = y3 + 2.0 * a;
  wi_a = y3 + 2.0 * a;
  xi_a = y3 + 2.0 * a;
  xi_a = (sqrt((b_y1 * b_y1 + y2 * y2) + xi_a * xi_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  yi_a = y3 + 2.0 * a;
  aj_a = y3 + 2.0 * a;
  bj_a = y3 + 2.0 * a;
  bj_a = (sqrt((b_y1 * b_y1 + y2 * y2) + bj_a * bj_a) + y3) + 2.0 * a;
  cj_a = y3 + 2.0 * a;
  dj_a = y3 + 2.0 * a;
  ej_a = y3 + 2.0 * a;
  fj_a = y3 + 2.0 * a;
  gj_a = y3 + 2.0 * a;
  hj_a = y3 + 2.0 * a;
  ij_a = y3 + 2.0 * a;
  ij_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ij_a * ij_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  jj_a = y3 + 2.0 * a;
  kj_a = y3 + 2.0 * a;
  lj_a = y3 + 2.0 * a;
  mj_a = y3 + 2.0 * a;
  nj_a = y3 + 2.0 * a;
  oj_a = y3 + 2.0 * a;
  pj_a = y3 + 2.0 * a;
  qj_a = y3 + 2.0 * a;
  rj_a = y3 + 2.0 * a;
  sj_a = y3 + 2.0 * a;
  tj_a = y3 + 2.0 * a;
  uj_a = y3 + 2.0 * a;
  uj_a = (b_y1 * b_y1 + y2 * y2) + uj_a * uj_a;
  vj_a = y3 + 2.0 * a;
  vj_a = (sqrt((b_y1 * b_y1 + y2 * y2) + vj_a * vj_a) + y3) + 2.0 * a;
  wj_a = y3 + 2.0 * a;
  xj_a = y3 + 2.0 * a;
  yj_a = y3 + 2.0 * a;
  ak_a = y3 + 2.0 * a;
  bk_a = y3 + 2.0 * a;
  ck_a = y3 + 2.0 * a;
  dk_a = y3 + 2.0 * a;
  ek_a = y3 + 2.0 * a;
  fk_a = y3 + 2.0 * a;
  gk_a = y3 + 2.0 * a;
  hk_a = y3 + 2.0 * a;
  ik_a = y3 + 2.0 * a;
  jk_a = y3 + 2.0 * a;
  kk_a = y3 + 2.0 * a;
  lk_a = y3 + 2.0 * a;
  lk_a = (sqrt((b_y1 * b_y1 + y2 * y2) + lk_a * lk_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  mk_a = y3 + 2.0 * a;
  nk_a = y3 + 2.0 * a;
  ok_a = y3 + 2.0 * a;
  pk_a = y3 + 2.0 * a;
  qk_a = y3 + 2.0 * a;
  rk_a = y3 + 2.0 * a;
  sk_a = y3 + 2.0 * a;
  tk_a = y3 + 2.0 * a;
  uk_a = y3 + 2.0 * a;
  vk_a = y3 + 2.0 * a;
  wk_a = y3 + 2.0 * a;
  wk_a = (sqrt((b_y1 * b_y1 + y2 * y2) + wk_a * wk_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  xk_a = y3 + 2.0 * a;
  yk_a = y3 + 2.0 * a;
  al_a = y3 + 2.0 * a;
  bl_a = y3 + 2.0 * a;
  cl_a = y3 + 2.0 * a;
  dl_a = y3 + 2.0 * a;
  dl_a = (b_y1 * b_y1 + y2 * y2) + dl_a * dl_a;
  el_a = y3 + 2.0 * a;
  fl_a = y3 + 2.0 * a;
  gl_a = y3 + 2.0 * a;
  hl_a = y3 + 2.0 * a;
  il_a = sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3;
  jl_a = y3 + 2.0 * a;
  kl_a = y3 + 2.0 * a;
  ll_a = y3 + 2.0 * a;
  ml_a = y3 + 2.0 * a;
  ml_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ml_a * ml_a) + y3) + 2.0 * a;
  nl_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  ol_a = y3 + 2.0 * a;
  pl_a = y3 + 2.0 * a;
  ql_a = y3 + 2.0 * a;
  rl_a = y3 + 2.0 * a;
  rl_a = (sqrt((b_y1 * b_y1 + y2 * y2) + rl_a * rl_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  sl_a = y3 + 2.0 * a;
  g_x = 1.0 / tan(b);
  tl_a = y3 + 2.0 * a;
  ul_a = y3 + 2.0 * a;
  h_x = 1.0 / tan(b);
  vl_a = y3 + 2.0 * a;
  wl_a = y3 + 2.0 * a;
  xl_a = y3 + 2.0 * a;
  xl_a = (sqrt((b_y1 * b_y1 + y2 * y2) + xl_a * xl_a) + y3) + 2.0 * a;
  yl_a = y3 + 2.0 * a;
  am_a = y3 + 2.0 * a;
  bm_a = y3 + 2.0 * a;
  cm_a = y3 + 2.0 * a;
  dm_a = y3 + 2.0 * a;
  em_a = y3 + 2.0 * a;
  fm_a = y3 + 2.0 * a;
  gm_a = y3 + 2.0 * a;
  gm_a = (sqrt((b_y1 * b_y1 + y2 * y2) + gm_a * gm_a) + y3) + 2.0 * a;
  hm_a = y3 + 2.0 * a;
  im_a = y3 + 2.0 * a;
  jm_a = y3 + 2.0 * a;
  km_a = y3 + 2.0 * a;
  lm_a = y3 + 2.0 * a;
  mm_a = y3 + 2.0 * a;
  nm_a = y3 + 2.0 * a;
  nm_a = (sqrt((b_y1 * b_y1 + y2 * y2) + nm_a * nm_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  om_a = y3 + 2.0 * a;
  pm_a = y3 + 2.0 * a;
  qm_a = y3 + 2.0 * a;
  rm_a = y3 + 2.0 * a;
  sm_a = y3 + 2.0 * a;
  tm_a = y3 + 2.0 * a;
  um_a = y3 + 2.0 * a;
  um_a = (sqrt((b_y1 * b_y1 + y2 * y2) + um_a * um_a) + y3) + 2.0 * a;
  vm_a = y3 + 2.0 * a;
  wm_a = y3 + 2.0 * a;
  xm_a = y3 + 2.0 * a;
  ym_a = y3 + 2.0 * a;
  an_a = y3 + 2.0 * a;
  bn_a = y3 + 2.0 * a;
  cn_a = y3 + 2.0 * a;
  dn_a = y3 + 2.0 * a;
  en_a = y3 + 2.0 * a;
  fn_a = y3 + 2.0 * a;
  gn_a = y3 + 2.0 * a;
  hn_a = y3 + 2.0 * a;
  in_a = y3 + 2.0 * a;
  jn_a = y3 + 2.0 * a;
  jn_a = (sqrt((b_y1 * b_y1 + y2 * y2) + jn_a * jn_a) + y3) + 2.0 * a;
  kn_a = y3 + 2.0 * a;
  ln_a = y3 + 2.0 * a;
  ln_a = (b_y1 * b_y1 + y2 * y2) + ln_a * ln_a;
  mn_a = y3 + 2.0 * a;
  nn_a = y3 + 2.0 * a;
  on_a = y3 + 2.0 * a;
  on_a = (sqrt((b_y1 * b_y1 + y2 * y2) + on_a * on_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  i_x = cos(b);
  pn_a = y3 + 2.0 * a;
  qn_a = y3 + 2.0 * a;
  rn_a = y3 + 2.0 * a;
  sn_a = y3 + 2.0 * a;
  j_x = cos(b);
  tn_a = y3 + 2.0 * a;
  un_a = y3 + 2.0 * a;
  vn_a = y3 + 2.0 * a;
  wn_a = y3 + 2.0 * a;
  xn_a = y3 + 2.0 * a;
  yn_a = y3 + 2.0 * a;
  ao_a = y3 + 2.0 * a;
  bo_a = y3 + 2.0 * a;
  co_a = y3 + 2.0 * a;
  do_a = y3 + 2.0 * a;
  k_x = cos(b);
  eo_a = y3 + 2.0 * a;
  fo_a = y3 + 2.0 * a;
  go_a = y3 + 2.0 * a;
  ho_a = y3 + 2.0 * a;
  ho_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ho_a * ho_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  l_x = cos(b);
  io_a = y3 + 2.0 * a;
  jo_a = y3 + 2.0 * a;
  ko_a = y3 + 2.0 * a;
  lo_a = y3 + 2.0 * a;
  mo_a = y3 + 2.0 * a;
  no_a = y3 + 2.0 * a;
  oo_a = y3 + 2.0 * a;
  po_a = y3 + 2.0 * a;
  qo_a = y3 + 2.0 * a;
  ro_a = y3 + 2.0 * a;
  so_a = b_y1 * cos(b) - y3 * sin(b);
  to_a = b_y1 * cos(b) - y3 * sin(b);
  uo_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  m_x = sin(b);
  vo_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  wo_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  xo_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  yo_a = y3 + 2.0 * a;
  ap_a = y3 + 2.0 * a;
  bp_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  cp_a = y3 + 2.0 * a;
  n_x = sin(b);
  dp_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  ep_a = y3 + 2.0 * a;
  fp_a = y3 + 2.0 * a;
  gp_a = sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3;
  hp_a = y3 + 2.0 * a;
  ip_a = y3 + 2.0 * a;
  jp_a = y3 + 2.0 * a;
  kp_a = y3 + 2.0 * a;
  kp_a = (sqrt((b_y1 * b_y1 + y2 * y2) + kp_a * kp_a) + y3) + 2.0 * a;
  lp_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  mp_a = y3 + 2.0 * a;
  np_a = y3 + 2.0 * a;
  op_a = y3 + 2.0 * a;
  pp_a = y3 + 2.0 * a;
  qp_a = y3 + 2.0 * a;
  rp_a = y3 + 2.0 * a;
  rp_a = (sqrt((b_y1 * b_y1 + y2 * y2) + rp_a * rp_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  sp_a = y3 + 2.0 * a;
  tp_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  up_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  vp_a = y3 + 2.0 * a;
  wp_a = y3 + 2.0 * a;
  xp_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  yp_a = y3 + 2.0 * a;
  o_x = sin(b);
  aq_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  p_x = 1.0 / tan(b);
  bq_a = y3 + 2.0 * a;
  bq_a = (sqrt((b_y1 * b_y1 + y2 * y2) + bq_a * bq_a) + y3) + 2.0 * a;
  cq_a = y3 + 2.0 * a;
  dq_a = y3 + 2.0 * a;
  eq_a = y3 + 2.0 * a;
  fq_a = y3 + 2.0 * a;
  gq_a = y3 + 2.0 * a;
  hq_a = y3 + 2.0 * a;
  iq_a = y3 + 2.0 * a;
  jq_a = y3 + 2.0 * a;
  kq_a = y3 + 2.0 * a;
  kq_a = (sqrt((b_y1 * b_y1 + y2 * y2) + kq_a * kq_a) + y3) + 2.0 * a;
  lq_a = y3 + 2.0 * a;
  mq_a = y3 + 2.0 * a;
  nq_a = y3 + 2.0 * a;
  oq_a = y3 + 2.0 * a;
  pq_a = y3 + 2.0 * a;
  pq_a = (sqrt((b_y1 * b_y1 + y2 * y2) + pq_a * pq_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  qq_a = y3 + 2.0 * a;
  rq_a = y3 + 2.0 * a;
  sq_a = y3 + 2.0 * a;
  tq_a = y3 + 2.0 * a;
  uq_a = y3 + 2.0 * a;
  vq_a = y3 + 2.0 * a;
  wq_a = y3 + 2.0 * a;
  xq_a = y3 + 2.0 * a;
  yq_a = y3 + 2.0 * a;
  ar_a = y3 + 2.0 * a;
  br_a = y3 + 2.0 * a;
  cr_a = y3 + 2.0 * a;
  dr_a = y3 + 2.0 * a;
  dr_a = (sqrt((b_y1 * b_y1 + y2 * y2) + dr_a * dr_a) + y3) + 2.0 * a;
  er_a = y3 + 2.0 * a;
  fr_a = y3 + 2.0 * a;
  gr_a = y3 + 2.0 * a;
  hr_a = y3 + 2.0 * a;
  ir_a = y3 + 2.0 * a;
  jr_a = y3 + 2.0 * a;
  kr_a = y3 + 2.0 * a;
  lr_a = y3 + 2.0 * a;
  lr_a = (sqrt((b_y1 * b_y1 + y2 * y2) + lr_a * lr_a) + y3) + 2.0 * a;
  mr_a = y3 + 2.0 * a;
  nr_a = y3 + 2.0 * a;
  or_a = y3 + 2.0 * a;
  pr_a = y3 + 2.0 * a;
  qr_a = y3 + 2.0 * a;
  rr_a = y3 + 2.0 * a;
  sr_a = y3 + 2.0 * a;
  tr_a = y3 + 2.0 * a;
  ur_a = y3 + 2.0 * a;
  vr_a = y3 + 2.0 * a;
  vr_a = (sqrt((b_y1 * b_y1 + y2 * y2) + vr_a * vr_a) + y3) + 2.0 * a;
  wr_a = y3 + 2.0 * a;
  xr_a = y3 + 2.0 * a;
  yr_a = y3 + 2.0 * a;
  as_a = y3 + 2.0 * a;
  bs_a = y3 + 2.0 * a;
  cs_a = y3 + 2.0 * a;
  ds_a = y3 + 2.0 * a;
  es_a = y3 + 2.0 * a;
  fs_a = y3 + 2.0 * a;
  fs_a = (sqrt((b_y1 * b_y1 + y2 * y2) + fs_a * fs_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  gs_a = y3 + 2.0 * a;
  hs_a = y3 + 2.0 * a;
  is_a = y3 + 2.0 * a;
  js_a = y3 + 2.0 * a;
  ks_a = y3 + 2.0 * a;
  ls_a = y3 + 2.0 * a;
  ms_a = y3 + 2.0 * a;
  ns_a = y3 + 2.0 * a;
  os_a = y3 + 2.0 * a;
  ps_a = y3 + 2.0 * a;
  qs_a = y3 + 2.0 * a;
  rs_a = y3 + 2.0 * a;
  rs_a = (sqrt((b_y1 * b_y1 + y2 * y2) + rs_a * rs_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  ss_a = y3 + 2.0 * a;
  ts_a = y3 + 2.0 * a;
  us_a = y3 + 2.0 * a;
  vs_a = y3 + 2.0 * a;
  ws_a = y3 + 2.0 * a;
  xs_a = y3 + 2.0 * a;
  xs_a = (b_y1 * b_y1 + y2 * y2) + xs_a * xs_a;
  ys_a = y3 + 2.0 * a;
  at_a = y3 + 2.0 * a;
  bt_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  ct_a = y3 + 2.0 * a;
  dt_a = y3 + 2.0 * a;
  et_a = y3 + 2.0 * a;
  ft_a = y3 + 2.0 * a;
  ft_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ft_a * ft_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  gt_a = y3 + 2.0 * a;
  ht_a = y3 + 2.0 * a;
  it_a = y3 + 2.0 * a;
  jt_a = y3 + 2.0 * a;
  kt_a = y3 + 2.0 * a;
  lt_a = y3 + 2.0 * a;
  lt_a = (sqrt((b_y1 * b_y1 + y2 * y2) + lt_a * lt_a) + y3) + 2.0 * a;
  mt_a = y3 + 2.0 * a;
  nt_a = y3 + 2.0 * a;
  ot_a = y3 + 2.0 * a;
  pt_a = y3 + 2.0 * a;
  qt_a = y3 + 2.0 * a;
  rt_a = y3 + 2.0 * a;
  st_a = y3 + 2.0 * a;
  st_a = (sqrt((b_y1 * b_y1 + y2 * y2) + st_a * st_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  tt_a = y3 + 2.0 * a;
  ut_a = y3 + 2.0 * a;
  vt_a = y3 + 2.0 * a;
  wt_a = y3 + 2.0 * a;
  xt_a = y3 + 2.0 * a;
  yt_a = y3 + 2.0 * a;
  au_a = y3 + 2.0 * a;
  bu_a = y3 + 2.0 * a;
  cu_a = y3 + 2.0 * a;
  du_a = y3 + 2.0 * a;
  eu_a = y3 + 2.0 * a;
  fu_a = y3 + 2.0 * a;
  fu_a = (b_y1 * b_y1 + y2 * y2) + fu_a * fu_a;
  gu_a = y3 + 2.0 * a;
  gu_a = (sqrt((b_y1 * b_y1 + y2 * y2) + gu_a * gu_a) + y3) + 2.0 * a;
  hu_a = y3 + 2.0 * a;
  iu_a = y3 + 2.0 * a;
  iu_a = (sqrt((b_y1 * b_y1 + y2 * y2) + iu_a * iu_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  ju_a = y3 + 2.0 * a;
  ku_a = y3 + 2.0 * a;
  lu_a = y3 + 2.0 * a;
  mu_a = y3 + 2.0 * a;
  nu_a = y3 + 2.0 * a;
  ou_a = y3 + 2.0 * a;
  pu_a = y3 + 2.0 * a;
  qu_a = y3 + 2.0 * a;
  ru_a = y3 + 2.0 * a;
  su_a = y3 + 2.0 * a;
  tu_a = y3 + 2.0 * a;
  uu_a = y3 + 2.0 * a;
  vu_a = y3 + 2.0 * a;
  wu_a = y3 + 2.0 * a;
  xu_a = y3 + 2.0 * a;
  yu_a = y3 + 2.0 * a;
  av_a = y3 + 2.0 * a;
  bv_a = y3 + 2.0 * a;
  cv_a = y3 + 2.0 * a;
  dv_a = y3 + 2.0 * a;
  ev_a = y3 + 2.0 * a;
  ev_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ev_a * ev_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  fv_a = y3 + 2.0 * a;
  gv_a = y3 + 2.0 * a;
  hv_a = y3 + 2.0 * a;
  iv_a = y3 + 2.0 * a;
  jv_a = y3 + 2.0 * a;
  kv_a = y3 + 2.0 * a;
  lv_a = y3 + 2.0 * a;
  mv_a = y3 + 2.0 * a;
  nv_a = y3 + 2.0 * a;
  ov_a = y3 + 2.0 * a;
  *e12 = ((((0.5 * B1 * (0.125 * (((((2.0 - 2.0 * nu) * ((((-2.0 / b_y1 / (1.0 +
    y2 * y2 / (b_y1 * b_y1)) + 1.0 / (b_y1 * cos(b) - y3 * sin(b)) / (1.0 + y2 *
    y2 / (b_a * b_a))) + ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) /
    (b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b)) + y2 * y2 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) / (b_y1 * (b_y1 * cos(b) - y3 *
    sin(b)) + y2 * y2 * cos(b))) - 2.0 * (y2 * y2) * sqrt((b_y1 * b_y1 + y2 * y2)
    + y3 * y3) * sin(b) / (c_a * c_a) * cos(b)) / (1.0 + y2 * y2 * ((b_y1 * b_y1
    + y2 * y2) + y3 * y3) * (x * x) / (d_a * d_a))) + 1.0 / (b_y1 * cos(b) + (y3
    + 2.0 * a) * sin(b)) / (1.0 + y2 * y2 / (e_a * e_a))) + ((sqrt((b_y1 * b_y1
    + y2 * y2) + f_a * f_a) * sin(b) / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) *
    sin(b)) + y2 * y2 * cos(b)) + y2 * y2 / sqrt((b_y1 * b_y1 + y2 * y2) + g_a *
    g_a) * sin(b) / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 *
                     cos(b))) - 2.0 * (y2 * y2) * sqrt((b_y1 * b_y1 + y2 * y2) +
    h_a * h_a) * sin(b) / (i_a * i_a) * cos(b)) / (1.0 + y2 * y2 * ((b_y1 * b_y1
    + y2 * y2) + j_a * j_a) * (b_x * b_x) / (k_a * k_a))) - b_y1 * (1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (sqrt((b_y1 * b_y1 + y2 * y2) + y3 *
    y3) - y3) + 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + l_a * l_a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + m_a * m_a) + y3) + 2.0 * a))) - b_y1 * y2 * (((-1.0 /
    powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / (sqrt((b_y1 * b_y1 +
    y2 * y2) + y3 * y3) - y3) * y2 - 1.0 / ((b_y1 * b_y1 + y2 * y2) + y3 * y3) /
    (n_a * n_a) * y2) - 1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + o_a * o_a,
    1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + p_a * p_a) + y3) + 2.0 * a) * y2) -
    1.0 / ((b_y1 * b_y1 + y2 * y2) + q_a * q_a) / (r_a * r_a) * y2)) - cos(b) *
    ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) - b_y1) / sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) -
    b_y1 * sin(b)) - y3 * cos(b)) + (sqrt((b_y1 * b_y1 + y2 * y2) + s_a * s_a) *
    sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + t_a * t_a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + u_a * u_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)))) -
    y2 * cos(b) * (((((1.0 / ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) * y2 /
                       ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b))
                        - y3 * cos(b)) - (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3)
    * sin(b) - b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) /
                       ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b))
                        - y3 * cos(b)) * y2) - (sqrt((b_y1 * b_y1 + y2 * y2) +
    y3 * y3) * sin(b) - b_y1) / ((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (v_a * v_a)
                      * y2) + 1.0 / ((b_y1 * b_y1 + y2 * y2) + w_a * w_a) * sin
                     (b) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + x_a * x_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) - (sqrt((b_y1 * b_y1 + y2 * y2) +
    y_a * y_a) * sin(b) - b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + ab_a *
    ab_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + bb_a * bb_a) - b_y1 * sin(b))
                  + (y3 + 2.0 * a) * cos(b)) * y2) - (sqrt((b_y1 * b_y1 + y2 *
    y2) + cb_a * cb_a) * sin(b) - b_y1) / ((b_y1 * b_y1 + y2 * y2) + db_a * db_a)
                   / (eb_a * eb_a) * y2)) / M_PI / (1.0 - nu) +
              0.25 * (((((((((((((((((-2.0 + 2.0 * nu) * (1.0 - 2.0 * nu) *
    ((-1.0 / b_y1 / (1.0 + y2 * y2 / (b_y1 * b_y1)) + 1.0 / (b_y1 * cos(b) + (y3
    + 2.0 * a) * sin(b)) / (1.0 + y2 * y2 / (fb_a * fb_a))) + ((sqrt((b_y1 *
    b_y1 + y2 * y2) + gb_a * gb_a) * sin(b) / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0
    * a) * sin(b)) + y2 * y2 * cos(b)) + y2 * y2 / sqrt((b_y1 * b_y1 + y2 * y2)
    + hb_a * hb_a) * sin(b) / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b))
    + y2 * y2 * cos(b))) - 2.0 * (y2 * y2) * sqrt((b_y1 * b_y1 + y2 * y2) + ib_a
    * ib_a) * sin(b) / (jb_a * jb_a) * cos(b)) / (1.0 + y2 * y2 * ((b_y1 * b_y1
    + y2 * y2) + kb_a * kb_a) * (c_x * c_x) / (lb_a * lb_a))) * (d_x * d_x) +
    (1.0 - 2.0 * nu) / ((sqrt((b_y1 * b_y1 + y2 * y2) + mb_a * mb_a) + y3) + 2.0
    * a) * (((1.0 - 2.0 * nu) - a / sqrt((b_y1 * b_y1 + y2 * y2) + nb_a * nb_a))
    * (1.0 / tan(b)) - b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + ob_a * ob_a) + y3)
    + 2.0 * a) * (nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + pb_a * pb_a)))) - (1.0
    - 2.0 * nu) * (y2 * y2) / (qb_a * qb_a) * (((1.0 - 2.0 * nu) - a / sqrt
    ((b_y1 * b_y1 + y2 * y2) + rb_a * rb_a)) * (1.0 / tan(b)) - b_y1 / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + sb_a * sb_a) + y3) + 2.0 * a) * (nu + a / sqrt
    ((b_y1 * b_y1 + y2 * y2) + tb_a * tb_a))) / sqrt((b_y1 * b_y1 + y2 * y2) +
    ub_a * ub_a)) + (1.0 - 2.0 * nu) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    vb_a * vb_a) + y3) + 2.0 * a) * ((a / powd_snf((b_y1 * b_y1 + y2 * y2) +
    wb_a * wb_a, 1.5) * y2 * (1.0 / tan(b)) + b_y1 / (xb_a * xb_a) * (nu + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + yb_a * yb_a)) / sqrt((b_y1 * b_y1 + y2 * y2)
    + ac_a * ac_a) * y2) + y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + bc_a * bc_a) +
    y3) + 2.0 * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + cc_a * cc_a, 1.5)
    * b_y1)) + (1.0 - 2.0 * nu) * cos(b) * (1.0 / tan(b)) / ((sqrt((b_y1 * b_y1
    + y2 * y2) + dc_a * dc_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos
    (b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + ec_a * ec_a))) - (1.0 - 2.0 * nu) *
    (y2 * y2) * cos(b) * (1.0 / tan(b)) / (fc_a * fc_a) * (cos(b) + a / sqrt
    ((b_y1 * b_y1 + y2 * y2) + gc_a * gc_a)) / sqrt((b_y1 * b_y1 + y2 * y2) +
    hc_a * hc_a)) - (1.0 - 2.0 * nu) * (y2 * y2) * cos(b) * (1.0 / tan(b)) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + ic_a * ic_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + jc_a * jc_a, 1.5))
    + a * (y3 + a) * (1.0 / tan(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + kc_a
    * kc_a, 1.5)) - 3.0 * a * (y2 * y2) * (y3 + a) * (1.0 / tan(b)) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + lc_a * lc_a, 2.5)) + (y3 + a) / sqrt
    ((b_y1 * b_y1 + y2 * y2) + mc_a * mc_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    nc_a * nc_a) + y3) + 2.0 * a) * (((-1.0 + 2.0 * nu) * (1.0 / tan(b)) + b_y1 /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + oc_a * oc_a) + y3) + 2.0 * a) * (2.0 * nu +
    a / sqrt((b_y1 * b_y1 + y2 * y2) + pc_a * pc_a))) + a * b_y1 / ((b_y1 * b_y1
    + y2 * y2) + qc_a * qc_a))) - y2 * y2 * (y3 + a) / powd_snf((b_y1 * b_y1
    + y2 * y2) + rc_a * rc_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + sc_a *
    sc_a) + y3) + 2.0 * a) * (((-1.0 + 2.0 * nu) * (1.0 / tan(b)) + b_y1 /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + tc_a * tc_a) + y3) + 2.0 * a) * (2.0 * nu +
    a / sqrt((b_y1 * b_y1 + y2 * y2) + uc_a * uc_a))) + a * b_y1 / ((b_y1 * b_y1
    + y2 * y2) + vc_a * vc_a))) - y2 * y2 * (y3 + a) / ((b_y1 * b_y1 + y2 * y2)
    + wc_a * wc_a) / (xc_a * xc_a) * (((-1.0 + 2.0 * nu) * (1.0 / tan(b)) + b_y1
    / ((sqrt((b_y1 * b_y1 + y2 * y2) + yc_a * yc_a) + y3) + 2.0 * a) * (2.0 * nu
    + a / sqrt((b_y1 * b_y1 + y2 * y2) + ad_a * ad_a))) + a * b_y1 / ((b_y1 *
    b_y1 + y2 * y2) + bd_a * bd_a))) + y2 * (y3 + a) / sqrt((b_y1 * b_y1 + y2 *
    y2) + cd_a * cd_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + dd_a * dd_a) + y3) +
    2.0 * a) * ((-b_y1 / (ed_a * ed_a) * (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 *
    y2) + fd_a * fd_a)) / sqrt((b_y1 * b_y1 + y2 * y2) + gd_a * gd_a) * y2 - y2 /
                 ((sqrt((b_y1 * b_y1 + y2 * y2) + hd_a * hd_a) + y3) + 2.0 * a) *
                 a / powd_snf((b_y1 * b_y1 + y2 * y2) + id_a * id_a, 1.5) *
                 b_y1) - 2.0 * a * b_y1 / (jd_a * jd_a) * y2)) + (y3 + a) / sqrt
    ((b_y1 * b_y1 + y2 * y2) + kd_a * kd_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    ld_a * ld_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + md_a * md_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b)) * (((sqrt((b_y1 * b_y1 + y2 * y2) + nd_a * nd_a) * cos(b) + y3) +
                2.0 * a) * ((1.0 - 2.0 * nu) * cos(b) - a / sqrt((b_y1 * b_y1 +
    y2 * y2) + od_a * od_a)) * (1.0 / tan(b)) + (2.0 - 2.0 * nu) * (sqrt((b_y1 *
    b_y1 + y2 * y2) + pd_a * pd_a) * sin(b) - b_y1) * cos(b)) - a * (y3 + 2.0 *
    a) * cos(b) * (1.0 / tan(b)) / ((b_y1 * b_y1 + y2 * y2) + qd_a * qd_a))) -
                        y2 * y2 * (y3 + a) / powd_snf((b_y1 * b_y1 + y2 * y2)
    + rd_a * rd_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + sd_a * sd_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + td_a * td_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((sqrt((b_y1 *
    b_y1 + y2 * y2) + ud_a * ud_a) * cos(b) + y3) + 2.0 * a) * ((1.0 - 2.0 * nu)
    * cos(b) - a / sqrt((b_y1 * b_y1 + y2 * y2) + vd_a * vd_a)) * (1.0 / tan(b))
    + (2.0 - 2.0 * nu) * (sqrt((b_y1 * b_y1 + y2 * y2) + wd_a * wd_a) * sin(b) -
    b_y1) * cos(b)) - a * (y3 + 2.0 * a) * cos(b) * (1.0 / tan(b)) / ((b_y1 *
    b_y1 + y2 * y2) + xd_a * xd_a))) - y2 * y2 * (y3 + a) / ((b_y1 * b_y1 + y2 *
    y2) + yd_a * yd_a) / (ae_a * ae_a) * (cos(b) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + be_a * be_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((sqrt((b_y1 *
    b_y1 + y2 * y2) + ce_a * ce_a) * cos(b) + y3) + 2.0 * a) * ((1.0 - 2.0 * nu)
    * cos(b) - a / sqrt((b_y1 * b_y1 + y2 * y2) + de_a * de_a)) * (1.0 / tan(b))
    + (2.0 - 2.0 * nu) * (sqrt((b_y1 * b_y1 + y2 * y2) + ee_a * ee_a) * sin(b) -
    b_y1) * cos(b)) - a * (y3 + 2.0 * a) * cos(b) * (1.0 / tan(b)) / ((b_y1 *
    b_y1 + y2 * y2) + fe_a * fe_a))) + y2 * (y3 + a) / sqrt((b_y1 * b_y1 + y2 *
    y2) + ge_a * ge_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + he_a * he_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((-cos(b) / (ie_a * ie_a) * (((sqrt
    ((b_y1 * b_y1 + y2 * y2) + je_a * je_a) * cos(b) + y3) + 2.0 * a) * ((1.0 -
    2.0 * nu) * cos(b) - a / sqrt((b_y1 * b_y1 + y2 * y2) + ke_a * ke_a)) * (1.0
    / tan(b)) + (2.0 - 2.0 * nu) * (sqrt((b_y1 * b_y1 + y2 * y2) + le_a * le_a) *
    sin(b) - b_y1) * cos(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + me_a * me_a) * y2
    + cos(b) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ne_a * ne_a) - b_y1 * sin(b)) +
                (y3 + 2.0 * a) * cos(b)) * ((1.0 / sqrt((b_y1 * b_y1 + y2 * y2)
    + oe_a * oe_a) * cos(b) * y2 * ((1.0 - 2.0 * nu) * cos(b) - a / sqrt((b_y1 *
    b_y1 + y2 * y2) + pe_a * pe_a)) * (1.0 / tan(b)) + ((sqrt((b_y1 * b_y1 + y2 *
    y2) + qe_a * qe_a) * cos(b) + y3) + 2.0 * a) * a / powd_snf((b_y1 * b_y1
    + y2 * y2) + re_a * re_a, 1.5) * y2 * (1.0 / tan(b))) + (2.0 - 2.0 * nu) /
    sqrt((b_y1 * b_y1 + y2 * y2) + se_a * se_a) * sin(b) * y2 * cos(b))) + 2.0 *
    a * (y3 + 2.0 * a) * cos(b) * (1.0 / tan(b)) / (te_a * te_a) * y2)) /
              M_PI / (1.0 - nu)) + 0.5 * B2 * (0.125 *
              ((((((((-1.0 + 2.0 * nu) * ((1.0 / sqrt((b_y1 * b_y1 + y2 * y2) +
    y3 * y3) * y2 / (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3) + 1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + ue_a * ue_a) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + ve_a * ve_a) + y3) + 2.0 * a)) - cos(b) * (1.0 / sqrt((b_y1 * b_y1 + y2 *
    y2) + y3 * y3) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin
    (b)) - y3 * cos(b)) + 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + we_a * we_a) * y2
    / ((sqrt((b_y1 * b_y1 + y2 * y2) + xe_a * xe_a) - b_y1 * sin(b)) + (y3 + 2.0
    * a) * cos(b)))) + b_y1 * b_y1 * (((-1.0 / powd_snf((b_y1 * b_y1 + y2 *
    y2) + y3 * y3, 1.5) / (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3) * y2 -
    1.0 / ((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (ye_a * ye_a) * y2) - 1.0 /
    powd_snf((b_y1 * b_y1 + y2 * y2) + af_a * af_a, 1.5) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + bf_a * bf_a) + y3) + 2.0 * a) * y2) - 1.0 / ((b_y1 * b_y1
    + y2 * y2) + cf_a * cf_a) / (df_a * df_a) * y2)) + (b_y1 * cos(b) - y3 * sin
    (b)) / ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) * y2 / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b))) - (b_y1 * cos(b)
    - y3 * sin(b)) * (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) - b_y1) /
                   powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) * y2) -
                  (b_y1 * cos(b) - y3 * sin(b)) * (sqrt((b_y1 * b_y1 + y2 * y2)
    + y3 * y3) * sin(b) - b_y1) / ((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (ef_a *
    ef_a) * y2) + (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / ((b_y1 * b_y1 + y2
    * y2) + ff_a * ff_a) * sin(b) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + gf_a *
    gf_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) - (b_y1 * cos(b) + (y3 +
    2.0 * a) * sin(b)) * (sqrt((b_y1 * b_y1 + y2 * y2) + hf_a * hf_a) * sin(b) -
    b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + if_a * if_a, 1.5) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + jf_a * jf_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b)) * y2) - (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (sqrt((b_y1 *
    b_y1 + y2 * y2) + kf_a * kf_a) * sin(b) - b_y1) / ((b_y1 * b_y1 + y2 * y2) +
    lf_a * lf_a) / (mf_a * mf_a) * y2) / M_PI / (1.0 - nu) + 0.25 *
              ((((((((((1.0 - 2.0 * nu) * (((2.0 - 2.0 * nu) * (e_x * e_x) + nu)
    / sqrt((b_y1 * b_y1 + y2 * y2) + nf_a * nf_a) * y2 / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + of_a * of_a) + y3) + 2.0 * a) - ((2.0 - 2.0 * nu) * (f_x * f_x) +
    1.0) * cos(b) / sqrt((b_y1 * b_y1 + y2 * y2) + pf_a * pf_a) * y2 / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + qf_a * qf_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b))) - (1.0 - 2.0 * nu) / (rf_a * rf_a) * (((((-1.0 + 2.0 * nu) * b_y1 *
    (1.0 / tan(b)) + nu * (y3 + 2.0 * a)) - a) + a * b_y1 * (1.0 / tan(b)) /
    sqrt((b_y1 * b_y1 + y2 * y2) + sf_a * sf_a)) + b_y1 * b_y1 / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + tf_a * tf_a) + y3) + 2.0 * a) * (nu + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + uf_a * uf_a))) / sqrt((b_y1 * b_y1 + y2 * y2) + vf_a *
    vf_a) * y2) + (1.0 - 2.0 * nu) / ((sqrt((b_y1 * b_y1 + y2 * y2) + wf_a *
    wf_a) + y3) + 2.0 * a) * ((-a * b_y1 * (1.0 / tan(b)) / powd_snf((b_y1 *
    b_y1 + y2 * y2) + xf_a * xf_a, 1.5) * y2 - b_y1 * b_y1 / (yf_a * yf_a) * (nu
    + a / sqrt((b_y1 * b_y1 + y2 * y2) + ag_a * ag_a)) / sqrt((b_y1 * b_y1 + y2 *
    y2) + bg_a * bg_a) * y2) - b_y1 * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    cg_a * cg_a) + y3) + 2.0 * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2) +
    dg_a * dg_a, 1.5) * y2)) + (1.0 - 2.0 * nu) * (1.0 / tan(b)) / (eg_a * eg_a)
                     * ((b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * cos(b) - a *
                        (sqrt((b_y1 * b_y1 + y2 * y2) + fg_a * fg_a) * sin(b) -
    b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + gg_a * gg_a) / cos(b)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + hg_a * hg_a) * y2) - (1.0 - 2.0 * nu) * (1.0 / tan(b)) /
                    ((sqrt((b_y1 * b_y1 + y2 * y2) + ig_a * ig_a) - b_y1 * sin(b))
                     + (y3 + 2.0 * a) * cos(b)) * (-a / ((b_y1 * b_y1 + y2 * y2)
    + jg_a * jg_a) * sin(b) * y2 / cos(b) + a * (sqrt((b_y1 * b_y1 + y2 * y2) +
    kg_a * kg_a) * sin(b) - b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + lg_a *
    lg_a, 1.5) / cos(b) * y2)) + 3.0 * a * y2 * (y3 + a) * (1.0 / tan(b)) /
                   powd_snf((b_y1 * b_y1 + y2 * y2) + mg_a * mg_a, 2.5) *
                   b_y1) - (y3 + a) / (ng_a * ng_a) * (((2.0 * nu + 1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + og_a * og_a) * ((1.0 - 2.0 * nu) * b_y1 * (1.0 /
    tan(b)) + a)) - b_y1 * b_y1 / sqrt((b_y1 * b_y1 + y2 * y2) + pg_a * pg_a) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + qg_a * qg_a) + y3) + 2.0 * a) * (2.0 * nu +
    a / sqrt((b_y1 * b_y1 + y2 * y2) + rg_a * rg_a))) - a * (b_y1 * b_y1) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + sg_a * sg_a, 1.5)) / sqrt((b_y1 * b_y1
    + y2 * y2) + tg_a * tg_a) * y2) + (y3 + a) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + ug_a * ug_a) + y3) + 2.0 * a) * ((((-1.0 / powd_snf((b_y1 * b_y1 + y2 *
    y2) + vg_a * vg_a, 1.5) * ((1.0 - 2.0 * nu) * b_y1 * (1.0 / tan(b)) + a) *
    y2 + b_y1 * b_y1 / powd_snf((b_y1 * b_y1 + y2 * y2) + wg_a * wg_a, 1.5) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + xg_a * xg_a) + y3) + 2.0 * a) * (2.0 * nu +
    a / sqrt((b_y1 * b_y1 + y2 * y2) + yg_a * yg_a)) * y2) + b_y1 * b_y1 /
    ((b_y1 * b_y1 + y2 * y2) + ah_a * ah_a) / (bh_a * bh_a) * (2.0 * nu + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + ch_a * ch_a)) * y2) + b_y1 * b_y1 / (dh_a *
    dh_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + eh_a * eh_a) + y3) + 2.0 * a) * a *
    y2) + 3.0 * a * (b_y1 * b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + fh_a *
    fh_a, 2.5) * y2)) - (y3 + a) * (1.0 / tan(b)) / (gh_a * gh_a) * ((-cos(b) *
    sin(b) + a * b_y1 * (y3 + 2.0 * a) / powd_snf((b_y1 * b_y1 + y2 * y2) +
    hh_a * hh_a, 1.5) / cos(b)) + (sqrt((b_y1 * b_y1 + y2 * y2) + ih_a * ih_a) *
    sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + jh_a * jh_a) * ((2.0 - 2.0 *
    nu) * cos(b) - ((sqrt((b_y1 * b_y1 + y2 * y2) + kh_a * kh_a) * cos(b) + y3)
                    + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + lh_a * lh_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt((b_y1 * b_y1 +
    y2 * y2) + mh_a * mh_a) / cos(b)))) / sqrt((b_y1 * b_y1 + y2 * y2) + nh_a *
    nh_a) * y2) + (y3 + a) * (1.0 / tan(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    oh_a * oh_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((-3.0 * a *
    b_y1 * (y3 + 2.0 * a) / powd_snf((b_y1 * b_y1 + y2 * y2) + ph_a * ph_a,
    2.5) / cos(b) * y2 + 1.0 / ((b_y1 * b_y1 + y2 * y2) + qh_a * qh_a) * sin(b) *
    y2 * ((2.0 - 2.0 * nu) * cos(b) - ((sqrt((b_y1 * b_y1 + y2 * y2) + rh_a *
    rh_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + sh_a *
    sh_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + th_a * th_a) / cos(b)))) - (sqrt((b_y1 * b_y1 + y2 * y2) +
    uh_a * uh_a) * sin(b) - b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + vh_a *
    vh_a, 1.5) * ((2.0 - 2.0 * nu) * cos(b) - ((sqrt((b_y1 * b_y1 + y2 * y2) +
    wh_a * wh_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    xh_a * xh_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt
    ((b_y1 * b_y1 + y2 * y2) + yh_a * yh_a) / cos(b))) * y2) + (sqrt((b_y1 *
    b_y1 + y2 * y2) + ai_a * ai_a) * sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 *
    y2) + bi_a * bi_a) * ((-1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + ci_a * ci_a) *
    cos(b) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + di_a * di_a) - b_y1 * sin(b))
                   + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt((b_y1 * b_y1 +
    y2 * y2) + ei_a * ei_a) / cos(b)) + ((sqrt((b_y1 * b_y1 + y2 * y2) + fi_a *
    fi_a) * cos(b) + y3) + 2.0 * a) / (gi_a * gi_a) * (1.0 + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + hi_a * hi_a) / cos(b)) / sqrt((b_y1 * b_y1 + y2 * y2) +
    ii_a * ii_a) * y2) + ((sqrt((b_y1 * b_y1 + y2 * y2) + ji_a * ji_a) * cos(b)
    + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ki_a * ki_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 * y2)
    + li_a * li_a, 1.5) / cos(b) * y2))) / M_PI / (1.0 - nu))) +
            0.5 * B3 * ((0.125 * sin(b) * ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 *
    y3) * sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) + (sqrt
    ((b_y1 * b_y1 + y2 * y2) + mi_a * mi_a) * sin(b) - b_y1) / sqrt((b_y1 * b_y1
    + y2 * y2) + ni_a * ni_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + oi_a * oi_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) / M_PI / (1.0 - nu)
              + 0.125 * y2 * sin(b) * (((((1.0 / ((b_y1 * b_y1 + y2 * y2) + y3 *
    y3) * sin(b) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b))
    - y3 * cos(b)) - (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) - b_y1) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) * y2) - (sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) * sin(b) - b_y1) / ((b_y1 * b_y1 + y2 * y2) + y3 *
    y3) / (pi_a * pi_a) * y2) + 1.0 / ((b_y1 * b_y1 + y2 * y2) + qi_a * qi_a) *
    sin(b) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + ri_a * ri_a) - b_y1 * sin(b))
                   + (y3 + 2.0 * a) * cos(b))) - (sqrt((b_y1 * b_y1 + y2 * y2) +
    si_a * si_a) * sin(b) - b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + ti_a *
    ti_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ui_a * ui_a) - b_y1 * sin(b))
                  + (y3 + 2.0 * a) * cos(b)) * y2) - (sqrt((b_y1 * b_y1 + y2 *
    y2) + vi_a * vi_a) * sin(b) - b_y1) / ((b_y1 * b_y1 + y2 * y2) + wi_a * wi_a)
    / (xi_a * xi_a) * y2) / M_PI / (1.0 - nu)) + 0.25 *
                        ((((((((1.0 - 2.0 * nu) * (((((1.0 / ((sqrt((b_y1 * b_y1
    + y2 * y2) + yi_a * yi_a) + y3) + 2.0 * a) * (1.0 + a / sqrt((b_y1 * b_y1 +
    y2 * y2) + aj_a * aj_a)) - y2 * y2 / (bj_a * bj_a) * (1.0 + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + cj_a * cj_a)) / sqrt((b_y1 * b_y1 + y2 * y2) + dj_a * dj_a))
    - y2 * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + ej_a * ej_a) + y3) + 2.0 * a) *
    a / powd_snf((b_y1 * b_y1 + y2 * y2) + fj_a * fj_a, 1.5)) - cos(b) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + gj_a * gj_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + hj_a * hj_a))) +
    y2 * y2 * cos(b) / (ij_a * ij_a) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2)
    + jj_a * jj_a)) / sqrt((b_y1 * b_y1 + y2 * y2) + kj_a * kj_a)) + y2 * y2 *
    cos(b) / ((sqrt((b_y1 * b_y1 + y2 * y2) + lj_a * lj_a) - b_y1 * sin(b)) +
              (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 * y2)
    + mj_a * mj_a, 1.5)) - (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) + nj_a * nj_a)
    * (a / ((b_y1 * b_y1 + y2 * y2) + oj_a * oj_a) + 1.0 / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + pj_a * pj_a) + y3) + 2.0 * a))) + y2 * y2 * (y3 + a) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + qj_a * qj_a, 1.5) * (a / ((b_y1 * b_y1
    + y2 * y2) + rj_a * rj_a) + 1.0 / ((sqrt((b_y1 * b_y1 + y2 * y2) + sj_a *
    sj_a) + y3) + 2.0 * a))) - y2 * (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) +
    tj_a * tj_a) * (-2.0 * a / (uj_a * uj_a) * y2 - 1.0 / (vj_a * vj_a) / sqrt
                    ((b_y1 * b_y1 + y2 * y2) + wj_a * wj_a) * y2)) + (y3 + a) *
    cos(b) / sqrt((b_y1 * b_y1 + y2 * y2) + xj_a * xj_a) / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + yj_a * yj_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    (((sqrt((b_y1 * b_y1 + y2 * y2) + ak_a * ak_a) * cos(b) + y3) + 2.0 * a) /
     ((sqrt((b_y1 * b_y1 + y2 * y2) + bk_a * bk_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + ck_a * ck_a)) +
     a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) + dk_a * dk_a))) - y2 * y2 *
    (y3 + a) * cos(b) / powd_snf((b_y1 * b_y1 + y2 * y2) + ek_a * ek_a, 1.5) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + fk_a * fk_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)) * (((sqrt((b_y1 * b_y1 + y2 * y2) + gk_a * gk_a) * cos(b) + y3)
                     + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + hk_a * hk_a)
    - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + ik_a * ik_a)) + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 *
    y2) + jk_a * jk_a))) - y2 * y2 * (y3 + a) * cos(b) / ((b_y1 * b_y1 + y2 * y2)
    + kk_a * kk_a) / (lk_a * lk_a) * (((sqrt((b_y1 * b_y1 + y2 * y2) + mk_a *
    mk_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + nk_a *
    nk_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1
    * b_y1 + y2 * y2) + ok_a * ok_a)) + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 *
    y2) + pk_a * pk_a))) + y2 * (y3 + a) * cos(b) / sqrt((b_y1 * b_y1 + y2 * y2)
    + qk_a * qk_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + rk_a * rk_a) - b_y1 * sin
                       (b)) + (y3 + 2.0 * a) * cos(b)) * (((1.0 / sqrt((b_y1 *
    b_y1 + y2 * y2) + sk_a * sk_a) * cos(b) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + tk_a * tk_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + uk_a * uk_a)) - ((sqrt((b_y1 * b_y1 + y2 * y2)
    + vk_a * vk_a) * cos(b) + y3) + 2.0 * a) / (wk_a * wk_a) * (cos(b) + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + xk_a * xk_a)) / sqrt((b_y1 * b_y1 + y2 * y2)
    + yk_a * yk_a) * y2) - ((sqrt((b_y1 * b_y1 + y2 * y2) + al_a * al_a) * cos(b)
    + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + bl_a * bl_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 * y2)
    + cl_a * cl_a, 1.5) * y2) - 2.0 * a * (y3 + 2.0 * a) / (dl_a * dl_a) * y2)) /
                        M_PI / (1.0 - nu))) + 0.5 * B1 * (0.125 *
            ((1.0 - 2.0 * nu) * ((1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) *
    b_y1 / (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3) + 1.0 / sqrt((b_y1 *
    b_y1 + y2 * y2) + el_a * el_a) * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    fl_a * fl_a) + y3) + 2.0 * a)) - cos(b) * ((1.0 / sqrt((b_y1 * b_y1 + y2 *
    y2) + y3 * y3) * b_y1 - sin(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3)
    - b_y1 * sin(b)) - y3 * cos(b)) + (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + gl_a
    * gl_a) * b_y1 - sin(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + hl_a * hl_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)))) - y2 * y2 * ((((-1.0 /
    powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / (sqrt((b_y1 * b_y1 +
    y2 * y2) + y3 * y3) - y3) * b_y1 - 1.0 / ((b_y1 * b_y1 + y2 * y2) + y3 * y3)
    / (il_a * il_a) * b_y1) - 1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + jl_a *
    jl_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + kl_a * kl_a) + y3) + 2.0 * a) *
    b_y1) - 1.0 / ((b_y1 * b_y1 + y2 * y2) + ll_a * ll_a) / (ml_a * ml_a) * b_y1)
              - cos(b) * (((-1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3,
    1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos
            (b)) * b_y1 - 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (nl_a *
    nl_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * b_y1 - sin(b))) -
    1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + ol_a * ol_a, 1.5) / ((sqrt((b_y1
    * b_y1 + y2 * y2) + pl_a * pl_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))
    * b_y1) - 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + ql_a * ql_a) / (rl_a * rl_a) *
    (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + sl_a * sl_a) * b_y1 - sin(b))))) /
            M_PI / (1.0 - nu) + 0.25 * ((((((((((((1.0 - 2.0 * nu)
    * (((2.0 - 2.0 * nu) * (g_x * g_x) - nu) / sqrt((b_y1 * b_y1 + y2 * y2) +
    tl_a * tl_a) * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + ul_a * ul_a) + y3) +
    2.0 * a) - (((2.0 - 2.0 * nu) * (h_x * h_x) + 1.0) - 2.0 * nu) * cos(b) *
       (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + vl_a * vl_a) * b_y1 - sin(b)) /
       ((sqrt((b_y1 * b_y1 + y2 * y2) + wl_a * wl_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b))) + (1.0 - 2.0 * nu) / (xl_a * xl_a) * (((b_y1 * (1.0 /
    tan(b)) * ((1.0 - 2.0 * nu) - a / sqrt((b_y1 * b_y1 + y2 * y2) + yl_a * yl_a))
    + nu * (y3 + 2.0 * a)) - a) + y2 * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    am_a * am_a) + y3) + 2.0 * a) * (nu + a / sqrt((b_y1 * b_y1 + y2 * y2) +
    bm_a * bm_a))) / sqrt((b_y1 * b_y1 + y2 * y2) + cm_a * cm_a) * b_y1) - (1.0
    - 2.0 * nu) / ((sqrt((b_y1 * b_y1 + y2 * y2) + dm_a * dm_a) + y3) + 2.0 * a)
    * (((((1.0 - 2.0 * nu) - a / sqrt((b_y1 * b_y1 + y2 * y2) + em_a * em_a)) *
         (1.0 / tan(b)) + a * (b_y1 * b_y1) * (1.0 / tan(b)) / powd_snf((b_y1
    * b_y1 + y2 * y2) + fm_a * fm_a, 1.5)) - y2 * y2 / (gm_a * gm_a) * (nu + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + hm_a * hm_a)) / sqrt((b_y1 * b_y1 + y2 * y2)
    + im_a * im_a) * b_y1) - y2 * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + jm_a *
    jm_a) + y3) + 2.0 * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + km_a *
    km_a, 1.5) * b_y1)) - (1.0 - 2.0 * nu) * cos(b) * (1.0 / tan(b)) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + lm_a * lm_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b)) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + mm_a * mm_a))) + (1.0
    - 2.0 * nu) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) /
    (nm_a * nm_a) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + om_a * om_a)) *
    (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + pm_a * pm_a) * b_y1 - sin(b))) + (1.0
    - 2.0 * nu) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + qm_a * qm_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + rm_a * rm_a, 1.5) *
    b_y1) - a * (y3 + a) * (1.0 / tan(b)) / powd_snf((b_y1 * b_y1 + y2 * y2)
    + sm_a * sm_a, 1.5)) + 3.0 * a * (b_y1 * b_y1) * (y3 + a) * (1.0 / tan(b)) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + tm_a * tm_a, 2.5)) - (y3 + a) / (um_a *
    um_a) * (((-2.0 * nu + 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + vm_a * vm_a) *
               ((1.0 - 2.0 * nu) * b_y1 * (1.0 / tan(b)) - a)) + y2 * y2 / sqrt
              ((b_y1 * b_y1 + y2 * y2) + wm_a * wm_a) / ((sqrt((b_y1 * b_y1 + y2
    * y2) + xm_a * xm_a) + y3) + 2.0 * a) * (2.0 * nu + a / sqrt((b_y1 * b_y1 +
    y2 * y2) + ym_a * ym_a))) + a * (y2 * y2) / powd_snf((b_y1 * b_y1 + y2 *
    y2) + an_a * an_a, 1.5)) / sqrt((b_y1 * b_y1 + y2 * y2) + bn_a * bn_a) *
    b_y1) + (y3 + a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + cn_a * cn_a) + y3) + 2.0
                        * a) * (((((-1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) +
    dn_a * dn_a, 1.5) * ((1.0 - 2.0 * nu) * b_y1 * (1.0 / tan(b)) - a) * b_y1 +
    1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + en_a * en_a) * (1.0 - 2.0 * nu) * (1.0 /
    tan(b))) - y2 * y2 / powd_snf((b_y1 * b_y1 + y2 * y2) + fn_a * fn_a, 1.5)
    / ((sqrt((b_y1 * b_y1 + y2 * y2) + gn_a * gn_a) + y3) + 2.0 * a) * (2.0 * nu
    + a / sqrt((b_y1 * b_y1 + y2 * y2) + hn_a * hn_a)) * b_y1) - y2 * y2 /
    ((b_y1 * b_y1 + y2 * y2) + in_a * in_a) / (jn_a * jn_a) * (2.0 * nu + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + kn_a * kn_a)) * b_y1) - y2 * y2 / (ln_a *
    ln_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + mn_a * mn_a) + y3) + 2.0 * a) * a *
    b_y1) - 3.0 * a * (y2 * y2) / powd_snf((b_y1 * b_y1 + y2 * y2) + nn_a *
    nn_a, 2.5) * b_y1)) - (y3 + a) / (on_a * on_a) * (((i_x * i_x - 1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + pn_a * pn_a) * ((1.0 - 2.0 * nu) * (b_y1 * cos(b)
    + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) + a * cos(b))) + a * (y3 + 2.0 *
    a) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + qn_a * qn_a, 1.5)) - 1.0 / sqrt((b_y1 *
    b_y1 + y2 * y2) + rn_a * rn_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + sn_a *
    sn_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (y2 * y2 * (j_x * j_x) -
    a * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + tn_a * tn_a) * ((sqrt((b_y1 * b_y1 + y2 * y2) + un_a *
    un_a) * cos(b) + y3) + 2.0 * a))) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) +
    vn_a * vn_a) * b_y1 - sin(b))) + (y3 + a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    wn_a * wn_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((((((1.0 /
    powd_snf((b_y1 * b_y1 + y2 * y2) + xn_a * xn_a, 1.5) * ((1.0 - 2.0 * nu) *
    (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) + a * cos(b)) *
    b_y1 - 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + yn_a * yn_a) * (1.0 - 2.0 * nu) *
    cos(b) * (1.0 / tan(b))) + a * (y3 + 2.0 * a) * cos(b) * (1.0 / tan(b)) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + ao_a * ao_a, 1.5)) - 3.0 * a * (y3 +
    2.0 * a) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + bo_a * bo_a, 2.5) * b_y1) + 1.0 /
    powd_snf((b_y1 * b_y1 + y2 * y2) + co_a * co_a, 1.5) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + do_a * do_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    (y2 * y2 * (k_x * k_x) - a * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) *
     (1.0 / tan(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + eo_a * eo_a) * ((sqrt((b_y1
    * b_y1 + y2 * y2) + fo_a * fo_a) * cos(b) + y3) + 2.0 * a)) * b_y1) + 1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + go_a * go_a) / (ho_a * ho_a) * (y2 * y2 *
    (l_x * l_x) - a * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b))
    / sqrt((b_y1 * b_y1 + y2 * y2) + io_a * io_a) * ((sqrt((b_y1 * b_y1 + y2 *
    y2) + jo_a * jo_a) * cos(b) + y3) + 2.0 * a)) * (1.0 / sqrt((b_y1 * b_y1 +
    y2 * y2) + ko_a * ko_a) * b_y1 - sin(b))) - 1.0 / sqrt((b_y1 * b_y1 + y2 *
    y2) + lo_a * lo_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + mo_a * mo_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((-a * cos(b) * (1.0 / tan(b)) / sqrt
    ((b_y1 * b_y1 + y2 * y2) + no_a * no_a) * ((sqrt((b_y1 * b_y1 + y2 * y2) +
    oo_a * oo_a) * cos(b) + y3) + 2.0 * a) + a * (b_y1 * cos(b) + (y3 + 2.0 * a)
    * sin(b)) * (1.0 / tan(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + po_a *
    po_a, 1.5) * ((sqrt((b_y1 * b_y1 + y2 * y2) + qo_a * qo_a) * cos(b) + y3) +
                  2.0 * a) * b_y1) - a * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b))
    * (1.0 / tan(b)) / ((b_y1 * b_y1 + y2 * y2) + ro_a * ro_a) * cos(b) * b_y1)))
            / M_PI / (1.0 - nu))) + 0.5 * B2 * (0.125 * ((((2.0 -
    2.0 * nu) * ((((2.0 * y2 / (b_y1 * b_y1) / (1.0 + y2 * y2 / (b_y1 * b_y1)) -
                    y2 / (so_a * so_a) * cos(b) / (1.0 + y2 * y2 / (to_a * to_a)))
                   + (y2 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) /
                      (b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b)) *
                      b_y1 - y2 * sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin
                      (b) / (uo_a * uo_a) * (2.0 * b_y1 * cos(b) - y3 * sin(b)))
                   / (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * (m_x
    * m_x) / (vo_a * vo_a))) - y2 / (wo_a * wo_a) * cos(b) / (1.0 + y2 * y2 /
    (xo_a * xo_a))) + (y2 / sqrt((b_y1 * b_y1 + y2 * y2) + yo_a * yo_a) * sin(b)
                       / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 *
    y2 * cos(b)) * b_y1 - y2 * sqrt((b_y1 * b_y1 + y2 * y2) + ap_a * ap_a) * sin
                       (b) / (bp_a * bp_a) * (2.0 * b_y1 * cos(b) + (y3 + 2.0 *
    a) * sin(b))) / (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2) + cp_a * cp_a) *
                     (n_x * n_x) / (dp_a * dp_a))) + y2 * (1.0 / sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) / (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3)
    + 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + ep_a * ep_a) / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + fp_a * fp_a) + y3) + 2.0 * a))) + b_y1 * y2 * (((-1.0 /
    powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / (sqrt((b_y1 * b_y1 +
    y2 * y2) + y3 * y3) - y3) * b_y1 - 1.0 / ((b_y1 * b_y1 + y2 * y2) + y3 * y3)
    / (gp_a * gp_a) * b_y1) - 1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + hp_a *
    hp_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ip_a * ip_a) + y3) + 2.0 * a) *
    b_y1) - 1.0 / ((b_y1 * b_y1 + y2 * y2) + jp_a * jp_a) / (kp_a * kp_a) * b_y1))
            - y2 * (((((cos(b) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) /
                        ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b))
    - y3 * cos(b)) - (b_y1 * cos(b) - y3 * sin(b)) / powd_snf((b_y1 * b_y1 +
    y2 * y2) + y3 * y3, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 *
    sin(b)) - y3 * cos(b)) * b_y1) - (b_y1 * cos(b) - y3 * sin(b)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) / (lp_a * lp_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 *
    y2) + y3 * y3) * b_y1 - sin(b))) + cos(b) / sqrt((b_y1 * b_y1 + y2 * y2) +
    mp_a * mp_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + np_a * np_a) - b_y1 * sin(b))
                    + (y3 + 2.0 * a) * cos(b))) - (b_y1 * cos(b) + (y3 + 2.0 * a)
    * sin(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + op_a * op_a, 1.5) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + pp_a * pp_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b)) * b_y1) - (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + qp_a * qp_a) / (rp_a * rp_a) * (1.0 / sqrt((b_y1 * b_y1 +
    y2 * y2) + sp_a * sp_a) * b_y1 - sin(b)))) / M_PI / (1.0 - nu)
           + 0.25 * ((((((((((((2.0 - 2.0 * nu) * (1.0 - 2.0 * nu) * ((y2 /
    (b_y1 * b_y1) / (1.0 + y2 * y2 / (b_y1 * b_y1)) - y2 / (tp_a * tp_a) * cos(b)
    / (1.0 + y2 * y2 / (up_a * up_a))) + (y2 / sqrt((b_y1 * b_y1 + y2 * y2) +
    vp_a * vp_a) * sin(b) / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) +
    y2 * y2 * cos(b)) * b_y1 - y2 * sqrt((b_y1 * b_y1 + y2 * y2) + wp_a * wp_a) *
    sin(b) / (xp_a * xp_a) * (2.0 * b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b))) /
    (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2) + yp_a * yp_a) * (o_x * o_x) /
     (aq_a * aq_a))) * (p_x * p_x) - (1.0 - 2.0 * nu) * y2 / (bq_a * bq_a) * (((
    -1.0 + 2.0 * nu) + a / sqrt((b_y1 * b_y1 + y2 * y2) + cq_a * cq_a)) * (1.0 /
    tan(b)) + b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + dq_a * dq_a) + y3) + 2.0 *
                      a) * (nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + eq_a * eq_a)))
    / sqrt((b_y1 * b_y1 + y2 * y2) + fq_a * fq_a) * b_y1) + (1.0 - 2.0 * nu) *
    y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + gq_a * gq_a) + y3) + 2.0 * a) * (((-a /
    powd_snf((b_y1 * b_y1 + y2 * y2) + hq_a * hq_a, 1.5) * b_y1 * (1.0 / tan
    (b)) + 1.0 / ((sqrt((b_y1 * b_y1 + y2 * y2) + iq_a * iq_a) + y3) + 2.0 * a) *
    (nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + jq_a * jq_a))) - b_y1 * b_y1 /
    (kq_a * kq_a) * (nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + lq_a * lq_a)) /
    sqrt((b_y1 * b_y1 + y2 * y2) + mq_a * mq_a)) - b_y1 * b_y1 / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + nq_a * nq_a) + y3) + 2.0 * a) * a / powd_snf((b_y1 *
    b_y1 + y2 * y2) + oq_a * oq_a, 1.5))) + (1.0 - 2.0 * nu) * y2 * (1.0 / tan(b))
    / (pq_a * pq_a) * (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2) + qq_a * qq_a) /
                       cos(b)) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + rq_a *
    rq_a) * b_y1 - sin(b))) + (1.0 - 2.0 * nu) * y2 * (1.0 / tan(b)) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + sq_a * sq_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + tq_a * tq_a, 1.5) / cos
    (b) * b_y1) + 3.0 * a * y2 * (y3 + a) * (1.0 / tan(b)) / powd_snf((b_y1 *
    b_y1 + y2 * y2) + uq_a * uq_a, 2.5) * b_y1) - y2 * (y3 + a) / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + vq_a * vq_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + wq_a * wq_a) + y3) + 2.0 * a) * (((1.0 - 2.0 * nu) * (1.0 / tan(b)) - 2.0 *
    nu * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + xq_a * xq_a) + y3) + 2.0 * a))
    - a * b_y1 / sqrt((b_y1 * b_y1 + y2 * y2) + yq_a * yq_a) * (1.0 / sqrt((b_y1
    * b_y1 + y2 * y2) + ar_a * ar_a) + 1.0 / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    br_a * br_a) + y3) + 2.0 * a))) * b_y1) - y2 * (y3 + a) / ((b_y1 * b_y1 + y2
    * y2) + cr_a * cr_a) / (dr_a * dr_a) * (((1.0 - 2.0 * nu) * (1.0 / tan(b)) -
    2.0 * nu * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + er_a * er_a) + y3) + 2.0 *
                       a)) - a * b_y1 / sqrt((b_y1 * b_y1 + y2 * y2) + fr_a *
    fr_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + gr_a * gr_a) + 1.0 / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + hr_a * hr_a) + y3) + 2.0 * a))) * b_y1) + y2 *
                        (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) + ir_a * ir_a) /
                        ((sqrt((b_y1 * b_y1 + y2 * y2) + jr_a * jr_a) + y3) +
    2.0 * a) * ((((-2.0 * nu / ((sqrt((b_y1 * b_y1 + y2 * y2) + kr_a * kr_a) +
    y3) + 2.0 * a) + 2.0 * nu * (b_y1 * b_y1) / (lr_a * lr_a) / sqrt((b_y1 *
    b_y1 + y2 * y2) + mr_a * mr_a)) - a / sqrt((b_y1 * b_y1 + y2 * y2) + nr_a *
    nr_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + or_a * or_a) + 1.0 / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + pr_a * pr_a) + y3) + 2.0 * a))) + a * (b_y1 *
    b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + qr_a * qr_a, 1.5) * (1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + rr_a * rr_a) + 1.0 / ((sqrt((b_y1 * b_y1 + y2
    * y2) + sr_a * sr_a) + y3) + 2.0 * a))) - a * b_y1 / sqrt((b_y1 * b_y1 + y2 *
    y2) + tr_a * tr_a) * (-1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + ur_a *
    ur_a, 1.5) * b_y1 - 1.0 / (vr_a * vr_a) / sqrt((b_y1 * b_y1 + y2 * y2) +
    wr_a * wr_a) * b_y1))) - y2 * (y3 + a) * (1.0 / tan(b)) / powd_snf((b_y1 *
    b_y1 + y2 * y2) + xr_a * xr_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + yr_a *
    yr_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((-2.0 + 2.0 * nu) *
    cos(b) + ((sqrt((b_y1 * b_y1 + y2 * y2) + as_a * as_a) * cos(b) + y3) + 2.0 *
              a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + bs_a * bs_a) - b_y1 * sin(b))
                    + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt((b_y1 * b_y1 +
    y2 * y2) + cs_a * cs_a) / cos(b))) + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2
    * y2) + ds_a * ds_a) / cos(b)) * b_y1) - y2 * (y3 + a) * (1.0 / tan(b)) /
                      sqrt((b_y1 * b_y1 + y2 * y2) + es_a * es_a) / (fs_a * fs_a)
                      * (((-2.0 + 2.0 * nu) * cos(b) + ((sqrt((b_y1 * b_y1 + y2 *
    y2) + gs_a * gs_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + hs_a * hs_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt
    ((b_y1 * b_y1 + y2 * y2) + is_a * is_a) / cos(b))) + a * (y3 + 2.0 * a) /
              ((b_y1 * b_y1 + y2 * y2) + js_a * js_a) / cos(b)) * (1.0 / sqrt
              ((b_y1 * b_y1 + y2 * y2) + ks_a * ks_a) * b_y1 - sin(b))) + y2 *
                     (y3 + a) * (1.0 / tan(b)) / sqrt((b_y1 * b_y1 + y2 * y2) +
             ls_a * ls_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ms_a * ms_a) - b_y1
              * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((1.0 / sqrt((b_y1 * b_y1
    + y2 * y2) + ns_a * ns_a) * cos(b) * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    os_a * os_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt
    ((b_y1 * b_y1 + y2 * y2) + ps_a * ps_a) / cos(b)) - ((sqrt((b_y1 * b_y1 + y2
    * y2) + qs_a * qs_a) * cos(b) + y3) + 2.0 * a) / (rs_a * rs_a) * (1.0 + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + ss_a * ss_a) / cos(b)) * (1.0 / sqrt((b_y1 *
    b_y1 + y2 * y2) + ts_a * ts_a) * b_y1 - sin(b))) - ((sqrt((b_y1 * b_y1 + y2 *
    y2) + us_a * us_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + vs_a * vs_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf
              ((b_y1 * b_y1 + y2 * y2) + ws_a * ws_a, 1.5) / cos(b) * b_y1) -
             2.0 * a * (y3 + 2.0 * a) / (xs_a * xs_a) / cos(b) * b_y1)) /
           M_PI / (1.0 - nu))) + 0.5 * B3 * (0.125 * ((1.0 - 2.0 *
    nu) * sin(b) * ((1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * b_y1 - sin
                     (b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 *
    sin(b)) - y3 * cos(b)) + (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + ys_a * ys_a) *
    b_y1 - sin(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + at_a * at_a) - b_y1 * sin
                       (b)) + (y3 + 2.0 * a) * cos(b))) - y2 * y2 * sin(b) * (((
    -1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) * b_y1 - 1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (bt_a * bt_a) * (1.0 / sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) * b_y1 - sin(b))) - 1.0 / powd_snf((b_y1 *
    b_y1 + y2 * y2) + ct_a * ct_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + dt_a *
    dt_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * b_y1) - 1.0 / sqrt((b_y1
    * b_y1 + y2 * y2) + et_a * et_a) / (ft_a * ft_a) * (1.0 / sqrt((b_y1 * b_y1
    + y2 * y2) + gt_a * gt_a) * b_y1 - sin(b)))) / M_PI / (1.0 -
    nu) + 0.25 * ((((((1.0 - 2.0 * nu) * ((((((-sin(b) * (1.0 / sqrt((b_y1 *
    b_y1 + y2 * y2) + ht_a * ht_a) * b_y1 - sin(b)) / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + it_a * it_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) - 1.0 /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + jt_a * jt_a) + y3) + 2.0 * a) * (1.0 + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + kt_a * kt_a))) + b_y1 * b_y1 / (lt_a * lt_a) *
    (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2) + mt_a * mt_a)) / sqrt((b_y1 * b_y1
    + y2 * y2) + nt_a * nt_a)) + b_y1 * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    ot_a * ot_a) + y3) + 2.0 * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2) +
    pt_a * pt_a, 1.5)) + cos(b) / ((sqrt((b_y1 * b_y1 + y2 * y2) + qt_a * qt_a)
    - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + rt_a * rt_a))) - (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b))
    / (st_a * st_a) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + tt_a * tt_a))
    * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + ut_a * ut_a) * b_y1 - sin(b))) -
    (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    vt_a * vt_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + wt_a * wt_a, 1.5) * b_y1) + (y3 + a) / sqrt((b_y1
    * b_y1 + y2 * y2) + xt_a * xt_a) * (a / ((b_y1 * b_y1 + y2 * y2) + yt_a *
    yt_a) + 1.0 / ((sqrt((b_y1 * b_y1 + y2 * y2) + au_a * au_a) + y3) + 2.0 * a)))
                     - b_y1 * b_y1 * (y3 + a) / powd_snf((b_y1 * b_y1 + y2 *
    y2) + bu_a * bu_a, 1.5) * (a / ((b_y1 * b_y1 + y2 * y2) + cu_a * cu_a) + 1.0
    / ((sqrt((b_y1 * b_y1 + y2 * y2) + du_a * du_a) + y3) + 2.0 * a))) + b_y1 *
                    (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) + eu_a * eu_a) *
                    (-2.0 * a / (fu_a * fu_a) * b_y1 - 1.0 / (gu_a * gu_a) /
                     sqrt((b_y1 * b_y1 + y2 * y2) + hu_a * hu_a) * b_y1)) + (y3
    + a) / (iu_a * iu_a) * ((sin(b) * (cos(b) - a / sqrt((b_y1 * b_y1 + y2 * y2)
    + ju_a * ju_a)) + (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + ku_a * ku_a) * (1.0 + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 +
    y2 * y2) + lu_a * lu_a))) - 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + mu_a * mu_a)
    / ((sqrt((b_y1 * b_y1 + y2 * y2) + nu_a * nu_a) - b_y1 * sin(b)) + (y3 + 2.0
    * a) * cos(b)) * (y2 * y2 * cos(b) * sin(b) - a * (b_y1 * cos(b) + (y3 + 2.0
    * a) * sin(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + ou_a * ou_a) * ((sqrt((b_y1 *
    b_y1 + y2 * y2) + pu_a * pu_a) * cos(b) + y3) + 2.0 * a))) * (1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + qu_a * qu_a) * b_y1 - sin(b))) - (y3 + a) /
                  ((sqrt((b_y1 * b_y1 + y2 * y2) + ru_a * ru_a) - b_y1 * sin(b))
                   + (y3 + 2.0 * a) * cos(b)) * ((((((sin(b) * a / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + su_a * su_a, 1.5) * b_y1 + cos(b) / sqrt((b_y1 *
    b_y1 + y2 * y2) + tu_a * tu_a) * (1.0 + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 +
    y2 * y2) + uu_a * uu_a))) - (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + vu_a * vu_a, 1.5) * (1.0 + a * (y3 +
    2.0 * a) / ((b_y1 * b_y1 + y2 * y2) + wu_a * wu_a)) * b_y1) - 2.0 * (b_y1 *
    cos(b) + (y3 + 2.0 * a) * sin(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) +
    xu_a * xu_a, 2.5) * a * (y3 + 2.0 * a) * b_y1) + 1.0 / powd_snf((b_y1 *
    b_y1 + y2 * y2) + yu_a * yu_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + av_a *
    av_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (y2 * y2 * cos(b) * sin
    (b) - a * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / sqrt((b_y1 * b_y1 + y2
    * y2) + bv_a * bv_a) * ((sqrt((b_y1 * b_y1 + y2 * y2) + cv_a * cv_a) * cos(b)
    + y3) + 2.0 * a)) * b_y1) + 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + dv_a * dv_a)
    / (ev_a * ev_a) * (y2 * y2 * cos(b) * sin(b) - a * (b_y1 * cos(b) + (y3 +
    2.0 * a) * sin(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + fv_a * fv_a) * ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + gv_a * gv_a) * cos(b) + y3) + 2.0 * a)) * (1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + hv_a * hv_a) * b_y1 - sin(b))) - 1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + iv_a * iv_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    jv_a * jv_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((-a * cos(b) /
    sqrt((b_y1 * b_y1 + y2 * y2) + kv_a * kv_a) * ((sqrt((b_y1 * b_y1 + y2 * y2)
    + lv_a * lv_a) * cos(b) + y3) + 2.0 * a) + a * (b_y1 * cos(b) + (y3 + 2.0 *
    a) * sin(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + mv_a * mv_a, 1.5) *
    ((sqrt((b_y1 * b_y1 + y2 * y2) + nv_a * nv_a) * cos(b) + y3) + 2.0 * a) *
    b_y1) - a * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / ((b_y1 * b_y1 + y2 *
    y2) + ov_a * ov_a) * cos(b) * b_y1))) / M_PI / (1.0 - nu));
  b_a = b_y1 * cos(b) - y3 * sin(b);
  c_a = b_y1 * cos(b) - y3 * sin(b);
  x = sin(b);
  d_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  b_x = sin(b);
  e_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  f_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  g_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  h_a = y3 + 2.0 * a;
  i_a = y3 + 2.0 * a;
  c_x = sin(b);
  j_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  k_a = y3 + 2.0 * a;
  d_x = sin(b);
  l_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  m_a = sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3;
  n_a = y3 + 2.0 * a;
  o_a = y3 + 2.0 * a;
  p_a = y3 + 2.0 * a;
  q_a = y3 + 2.0 * a;
  q_a = (sqrt((b_y1 * b_y1 + y2 * y2) + q_a * q_a) + y3) + 2.0 * a;
  r_a = y3 + 2.0 * a;
  s_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  t_a = y3 + 2.0 * a;
  u_a = y3 + 2.0 * a;
  v_a = y3 + 2.0 * a;
  w_a = y3 + 2.0 * a;
  x_a = y3 + 2.0 * a;
  y_a = y3 + 2.0 * a;
  ab_a = y3 + 2.0 * a;
  bb_a = y3 + 2.0 * a;
  bb_a = (sqrt((b_y1 * b_y1 + y2 * y2) + bb_a * bb_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  cb_a = y3 + 2.0 * a;
  db_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  eb_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  fb_a = y3 + 2.0 * a;
  gb_a = y3 + 2.0 * a;
  e_x = sin(b);
  hb_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  ib_a = y3 + 2.0 * a;
  f_x = sin(b);
  jb_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  g_x = 1.0 / tan(b);
  kb_a = y3 + 2.0 * a;
  kb_a = (sqrt((b_y1 * b_y1 + y2 * y2) + kb_a * kb_a) + y3) + 2.0 * a;
  lb_a = y3 + 2.0 * a;
  mb_a = y3 + 2.0 * a;
  nb_a = y3 + 2.0 * a;
  ob_a = y3 + 2.0 * a;
  pb_a = y3 + 2.0 * a;
  qb_a = y3 + 2.0 * a;
  rb_a = y3 + 2.0 * a;
  rb_a = (sqrt((b_y1 * b_y1 + y2 * y2) + rb_a * rb_a) + y3) + 2.0 * a;
  sb_a = y3 + 2.0 * a;
  tb_a = y3 + 2.0 * a;
  ub_a = y3 + 2.0 * a;
  vb_a = y3 + 2.0 * a;
  wb_a = y3 + 2.0 * a;
  wb_a = (sqrt((b_y1 * b_y1 + y2 * y2) + wb_a * wb_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  xb_a = y3 + 2.0 * a;
  yb_a = y3 + 2.0 * a;
  ac_a = y3 + 2.0 * a;
  bc_a = y3 + 2.0 * a;
  cc_a = y3 + 2.0 * a;
  dc_a = y3 + 2.0 * a;
  ec_a = y3 + 2.0 * a;
  fc_a = y3 + 2.0 * a;
  gc_a = y3 + 2.0 * a;
  hc_a = y3 + 2.0 * a;
  ic_a = y3 + 2.0 * a;
  jc_a = y3 + 2.0 * a;
  kc_a = y3 + 2.0 * a;
  lc_a = y3 + 2.0 * a;
  mc_a = y3 + 2.0 * a;
  nc_a = y3 + 2.0 * a;
  oc_a = y3 + 2.0 * a;
  pc_a = y3 + 2.0 * a;
  pc_a = (sqrt((b_y1 * b_y1 + y2 * y2) + pc_a * pc_a) + y3) + 2.0 * a;
  qc_a = y3 + 2.0 * a;
  rc_a = y3 + 2.0 * a;
  sc_a = y3 + 2.0 * a;
  tc_a = y3 + 2.0 * a;
  uc_a = y3 + 2.0 * a;
  vc_a = y3 + 2.0 * a;
  wc_a = y3 + 2.0 * a;
  wc_a = (sqrt((b_y1 * b_y1 + y2 * y2) + wc_a * wc_a) + y3) + 2.0 * a;
  xc_a = y3 + 2.0 * a;
  yc_a = y3 + 2.0 * a;
  ad_a = y3 + 2.0 * a;
  bd_a = y3 + 2.0 * a;
  cd_a = y3 + 2.0 * a;
  cd_a = (b_y1 * b_y1 + y2 * y2) + cd_a * cd_a;
  dd_a = y3 + 2.0 * a;
  ed_a = y3 + 2.0 * a;
  fd_a = y3 + 2.0 * a;
  gd_a = y3 + 2.0 * a;
  hd_a = y3 + 2.0 * a;
  id_a = y3 + 2.0 * a;
  jd_a = y3 + 2.0 * a;
  kd_a = y3 + 2.0 * a;
  ld_a = y3 + 2.0 * a;
  md_a = y3 + 2.0 * a;
  nd_a = y3 + 2.0 * a;
  od_a = y3 + 2.0 * a;
  pd_a = y3 + 2.0 * a;
  qd_a = y3 + 2.0 * a;
  rd_a = y3 + 2.0 * a;
  sd_a = y3 + 2.0 * a;
  sd_a = (sqrt((b_y1 * b_y1 + y2 * y2) + sd_a * sd_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  td_a = y3 + 2.0 * a;
  ud_a = y3 + 2.0 * a;
  vd_a = y3 + 2.0 * a;
  wd_a = y3 + 2.0 * a;
  xd_a = y3 + 2.0 * a;
  yd_a = y3 + 2.0 * a;
  ae_a = y3 + 2.0 * a;
  be_a = y3 + 2.0 * a;
  ce_a = y3 + 2.0 * a;
  ce_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ce_a * ce_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  de_a = y3 + 2.0 * a;
  ee_a = y3 + 2.0 * a;
  fe_a = y3 + 2.0 * a;
  ge_a = y3 + 2.0 * a;
  he_a = y3 + 2.0 * a;
  ie_a = y3 + 2.0 * a;
  je_a = y3 + 2.0 * a;
  ke_a = y3 + 2.0 * a;
  le_a = y3 + 2.0 * a;
  me_a = y3 + 2.0 * a;
  ne_a = y3 + 2.0 * a;
  oe_a = y3 + 2.0 * a;
  oe_a = (b_y1 * b_y1 + y2 * y2) + oe_a * oe_a;
  pe_a = y3 + 2.0 * a;
  qe_a = y3 + 2.0 * a;
  re_a = y3 + 2.0 * a;
  se_a = y3 + 2.0 * a;
  te_a = sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3;
  ue_a = y3 + 2.0 * a;
  ve_a = y3 + 2.0 * a;
  we_a = y3 + 2.0 * a;
  xe_a = y3 + 2.0 * a;
  xe_a = (sqrt((b_y1 * b_y1 + y2 * y2) + xe_a * xe_a) + y3) + 2.0 * a;
  ye_a = y3 + 2.0 * a;
  af_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  bf_a = y3 + 2.0 * a;
  cf_a = y3 + 2.0 * a;
  df_a = y3 + 2.0 * a;
  ef_a = y3 + 2.0 * a;
  ff_a = y3 + 2.0 * a;
  gf_a = y3 + 2.0 * a;
  hf_a = y3 + 2.0 * a;
  if_a = y3 + 2.0 * a;
  jf_a = y3 + 2.0 * a;
  kf_a = y3 + 2.0 * a;
  lf_a = y3 + 2.0 * a;
  lf_a = (sqrt((b_y1 * b_y1 + y2 * y2) + lf_a * lf_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  mf_a = y3 + 2.0 * a;
  h_x = 1.0 / tan(b);
  nf_a = y3 + 2.0 * a;
  of_a = y3 + 2.0 * a;
  i_x = 1.0 / tan(b);
  pf_a = y3 + 2.0 * a;
  qf_a = y3 + 2.0 * a;
  rf_a = y3 + 2.0 * a;
  rf_a = (sqrt((b_y1 * b_y1 + y2 * y2) + rf_a * rf_a) + y3) + 2.0 * a;
  sf_a = y3 + 2.0 * a;
  tf_a = y3 + 2.0 * a;
  uf_a = y3 + 2.0 * a;
  vf_a = y3 + 2.0 * a;
  wf_a = y3 + 2.0 * a;
  xf_a = y3 + 2.0 * a;
  yf_a = y3 + 2.0 * a;
  yf_a = (sqrt((b_y1 * b_y1 + y2 * y2) + yf_a * yf_a) + y3) + 2.0 * a;
  ag_a = y3 + 2.0 * a;
  bg_a = y3 + 2.0 * a;
  cg_a = y3 + 2.0 * a;
  dg_a = y3 + 2.0 * a;
  eg_a = y3 + 2.0 * a;
  eg_a = (sqrt((b_y1 * b_y1 + y2 * y2) + eg_a * eg_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  fg_a = y3 + 2.0 * a;
  gg_a = y3 + 2.0 * a;
  hg_a = y3 + 2.0 * a;
  ig_a = y3 + 2.0 * a;
  jg_a = y3 + 2.0 * a;
  kg_a = y3 + 2.0 * a;
  lg_a = y3 + 2.0 * a;
  mg_a = y3 + 2.0 * a;
  ng_a = y3 + 2.0 * a;
  og_a = y3 + 2.0 * a;
  pg_a = y3 + 2.0 * a;
  qg_a = y3 + 2.0 * a;
  rg_a = y3 + 2.0 * a;
  sg_a = y3 + 2.0 * a;
  tg_a = y3 + 2.0 * a;
  ug_a = y3 + 2.0 * a;
  ug_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ug_a * ug_a) + y3) + 2.0 * a;
  vg_a = y3 + 2.0 * a;
  wg_a = y3 + 2.0 * a;
  xg_a = y3 + 2.0 * a;
  yg_a = y3 + 2.0 * a;
  ah_a = y3 + 2.0 * a;
  bh_a = y3 + 2.0 * a;
  ch_a = y3 + 2.0 * a;
  dh_a = y3 + 2.0 * a;
  eh_a = y3 + 2.0 * a;
  fh_a = y3 + 2.0 * a;
  gh_a = y3 + 2.0 * a;
  hh_a = y3 + 2.0 * a;
  ih_a = y3 + 2.0 * a;
  ih_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ih_a * ih_a) + y3) + 2.0 * a;
  jh_a = y3 + 2.0 * a;
  kh_a = y3 + 2.0 * a;
  lh_a = y3 + 2.0 * a;
  lh_a = (b_y1 * b_y1 + y2 * y2) + lh_a * lh_a;
  mh_a = y3 + 2.0 * a;
  nh_a = y3 + 2.0 * a;
  oh_a = y3 + 2.0 * a;
  ph_a = y3 + 2.0 * a;
  qh_a = y3 + 2.0 * a;
  rh_a = y3 + 2.0 * a;
  sh_a = y3 + 2.0 * a;
  th_a = y3 + 2.0 * a;
  uh_a = y3 + 2.0 * a;
  vh_a = y3 + 2.0 * a;
  vh_a = (sqrt((b_y1 * b_y1 + y2 * y2) + vh_a * vh_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  wh_a = y3 + 2.0 * a;
  xh_a = y3 + 2.0 * a;
  yh_a = y3 + 2.0 * a;
  ai_a = y3 + 2.0 * a;
  bi_a = y3 + 2.0 * a;
  ci_a = y3 + 2.0 * a;
  di_a = y3 + 2.0 * a;
  ei_a = y3 + 2.0 * a;
  fi_a = y3 + 2.0 * a;
  gi_a = y3 + 2.0 * a;
  hi_a = y3 + 2.0 * a;
  ii_a = y3 + 2.0 * a;
  ji_a = y3 + 2.0 * a;
  ki_a = y3 + 2.0 * a;
  li_a = y3 + 2.0 * a;
  mi_a = y3 + 2.0 * a;
  ni_a = y3 + 2.0 * a;
  oi_a = y3 + 2.0 * a;
  pi_a = y3 + 2.0 * a;
  qi_a = y3 + 2.0 * a;
  ri_a = y3 + 2.0 * a;
  si_a = y3 + 2.0 * a;
  ti_a = y3 + 2.0 * a;
  ui_a = y3 + 2.0 * a;
  vi_a = y3 + 2.0 * a;
  wi_a = y3 + 2.0 * a;
  wi_a = (sqrt((b_y1 * b_y1 + y2 * y2) + wi_a * wi_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  xi_a = y3 + 2.0 * a;
  yi_a = y3 + 2.0 * a;
  aj_a = y3 + 2.0 * a;
  bj_a = y3 + 2.0 * a;
  cj_a = y3 + 2.0 * a;
  dj_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  ej_a = y3 + 2.0 * a;
  fj_a = y3 + 2.0 * a;
  gj_a = y3 + 2.0 * a;
  hj_a = y3 + 2.0 * a;
  ij_a = y3 + 2.0 * a;
  jj_a = y3 + 2.0 * a;
  kj_a = y3 + 2.0 * a;
  lj_a = y3 + 2.0 * a;
  lj_a = (sqrt((b_y1 * b_y1 + y2 * y2) + lj_a * lj_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  mj_a = y3 + 2.0 * a;
  nj_a = y3 + 2.0 * a;
  nj_a = (sqrt((b_y1 * b_y1 + y2 * y2) + nj_a * nj_a) + y3) + 2.0 * a;
  oj_a = y3 + 2.0 * a;
  pj_a = y3 + 2.0 * a;
  qj_a = y3 + 2.0 * a;
  rj_a = y3 + 2.0 * a;
  sj_a = y3 + 2.0 * a;
  sj_a = (sqrt((b_y1 * b_y1 + y2 * y2) + sj_a * sj_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  tj_a = y3 + 2.0 * a;
  uj_a = y3 + 2.0 * a;
  vj_a = y3 + 2.0 * a;
  wj_a = y3 + 2.0 * a;
  xj_a = y3 + 2.0 * a;
  yj_a = y3 + 2.0 * a;
  ak_a = y3 + 2.0 * a;
  bk_a = y3 + 2.0 * a;
  ck_a = y3 + 2.0 * a;
  dk_a = y3 + 2.0 * a;
  ek_a = y3 + 2.0 * a;
  fk_a = y3 + 2.0 * a;
  fk_a = (b_y1 * b_y1 + y2 * y2) + fk_a * fk_a;
  gk_a = y3 + 2.0 * a;
  gk_a = (sqrt((b_y1 * b_y1 + y2 * y2) + gk_a * gk_a) + y3) + 2.0 * a;
  hk_a = y3 + 2.0 * a;
  ik_a = y3 + 2.0 * a;
  jk_a = y3 + 2.0 * a;
  kk_a = y3 + 2.0 * a;
  lk_a = y3 + 2.0 * a;
  mk_a = y3 + 2.0 * a;
  nk_a = y3 + 2.0 * a;
  ok_a = y3 + 2.0 * a;
  pk_a = y3 + 2.0 * a;
  qk_a = y3 + 2.0 * a;
  rk_a = y3 + 2.0 * a;
  sk_a = y3 + 2.0 * a;
  tk_a = y3 + 2.0 * a;
  uk_a = y3 + 2.0 * a;
  vk_a = y3 + 2.0 * a;
  vk_a = (sqrt((b_y1 * b_y1 + y2 * y2) + vk_a * vk_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  wk_a = y3 + 2.0 * a;
  xk_a = y3 + 2.0 * a;
  yk_a = y3 + 2.0 * a;
  al_a = y3 + 2.0 * a;
  bl_a = y3 + 2.0 * a;
  cl_a = y3 + 2.0 * a;
  dl_a = y3 + 2.0 * a;
  el_a = y3 + 2.0 * a;
  fl_a = y3 + 2.0 * a;
  gl_a = y3 + 2.0 * a;
  hl_a = y3 + 2.0 * a;
  il_a = y3 + 2.0 * a;
  il_a = (sqrt((b_y1 * b_y1 + y2 * y2) + il_a * il_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  jl_a = y3 + 2.0 * a;
  kl_a = y3 + 2.0 * a;
  ll_a = y3 + 2.0 * a;
  ml_a = y3 + 2.0 * a;
  nl_a = y3 + 2.0 * a;
  ol_a = y3 + 2.0 * a;
  pl_a = y3 + 2.0 * a;
  pl_a = (b_y1 * b_y1 + y2 * y2) + pl_a * pl_a;
  ql_a = y3 + 2.0 * a;
  rl_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  sl_a = y3 + 2.0 * a;
  tl_a = y3 + 2.0 * a;
  ul_a = y3 + 2.0 * a;
  vl_a = y3 + 2.0 * a;
  wl_a = y3 + 2.0 * a;
  xl_a = y3 + 2.0 * a;
  yl_a = y3 + 2.0 * a;
  am_a = y3 + 2.0 * a;
  am_a = (sqrt((b_y1 * b_y1 + y2 * y2) + am_a * am_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  bm_a = y3 + 2.0 * a;
  cm_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  dm_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  em_a = y3 + 2.0 * a;
  fm_a = y3 + 2.0 * a;
  gm_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  hm_a = y3 + 2.0 * a;
  j_x = sin(b);
  im_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  jm_a = y3 + 2.0 * a;
  jm_a = (sqrt((b_y1 * b_y1 + y2 * y2) + jm_a * jm_a) + y3) + 2.0 * a;
  km_a = y3 + 2.0 * a;
  lm_a = y3 + 2.0 * a;
  mm_a = y3 + 2.0 * a;
  nm_a = y3 + 2.0 * a;
  om_a = y3 + 2.0 * a;
  om_a = (sqrt((b_y1 * b_y1 + y2 * y2) + om_a * om_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  pm_a = y3 + 2.0 * a;
  qm_a = y3 + 2.0 * a;
  rm_a = y3 + 2.0 * a;
  sm_a = y3 + 2.0 * a;
  tm_a = y3 + 2.0 * a;
  um_a = y3 + 2.0 * a;
  vm_a = y3 + 2.0 * a;
  wm_a = y3 + 2.0 * a;
  xm_a = y3 + 2.0 * a;
  xm_a = (sqrt((b_y1 * b_y1 + y2 * y2) + xm_a * xm_a) + y3) + 2.0 * a;
  ym_a = y3 + 2.0 * a;
  an_a = y3 + 2.0 * a;
  an_a = (b_y1 * b_y1 + y2 * y2) + an_a * an_a;
  bn_a = y3 + 2.0 * a;
  cn_a = y3 + 2.0 * a;
  dn_a = y3 + 2.0 * a;
  en_a = y3 + 2.0 * a;
  fn_a = y3 + 2.0 * a;
  gn_a = y3 + 2.0 * a;
  hn_a = y3 + 2.0 * a;
  in_a = y3 + 2.0 * a;
  in_a = (sqrt((b_y1 * b_y1 + y2 * y2) + in_a * in_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  jn_a = y3 + 2.0 * a;
  kn_a = y3 + 2.0 * a;
  ln_a = y3 + 2.0 * a;
  mn_a = y3 + 2.0 * a;
  nn_a = y3 + 2.0 * a;
  on_a = y3 + 2.0 * a;
  pn_a = y3 + 2.0 * a;
  qn_a = y3 + 2.0 * a;
  rn_a = y3 + 2.0 * a;
  sn_a = y3 + 2.0 * a;
  tn_a = y3 + 2.0 * a;
  un_a = y3 + 2.0 * a;
  un_a = (sqrt((b_y1 * b_y1 + y2 * y2) + un_a * un_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  vn_a = y3 + 2.0 * a;
  wn_a = y3 + 2.0 * a;
  xn_a = y3 + 2.0 * a;
  yn_a = y3 + 2.0 * a;
  ao_a = y3 + 2.0 * a;
  bo_a = y3 + 2.0 * a;
  bo_a = (b_y1 * b_y1 + y2 * y2) + bo_a * bo_a;
  co_a = y3 + 2.0 * a;
  do_a = y3 + 2.0 * a;
  eo_a = y3 + 2.0 * a;
  fo_a = y3 + 2.0 * a;
  go_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  ho_a = y3 + 2.0 * a;
  io_a = y3 + 2.0 * a;
  jo_a = y3 + 2.0 * a;
  ko_a = y3 + 2.0 * a;
  lo_a = y3 + 2.0 * a;
  mo_a = y3 + 2.0 * a;
  no_a = y3 + 2.0 * a;
  oo_a = y3 + 2.0 * a;
  po_a = y3 + 2.0 * a;
  qo_a = y3 + 2.0 * a;
  ro_a = y3 + 2.0 * a;
  ro_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ro_a * ro_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  so_a = y3 + 2.0 * a;
  to_a = y3 + 2.0 * a;
  uo_a = y3 + 2.0 * a;
  vo_a = y3 + 2.0 * a;
  wo_a = y3 + 2.0 * a;
  xo_a = y3 + 2.0 * a;
  yo_a = y3 + 2.0 * a;
  ap_a = y3 + 2.0 * a;
  ap_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ap_a * ap_a) + y3) + 2.0 * a;
  bp_a = y3 + 2.0 * a;
  cp_a = y3 + 2.0 * a;
  dp_a = y3 + 2.0 * a;
  ep_a = y3 + 2.0 * a;
  fp_a = y3 + 2.0 * a;
  gp_a = y3 + 2.0 * a;
  hp_a = y3 + 2.0 * a;
  hp_a = (sqrt((b_y1 * b_y1 + y2 * y2) + hp_a * hp_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  ip_a = y3 + 2.0 * a;
  jp_a = y3 + 2.0 * a;
  kp_a = y3 + 2.0 * a;
  lp_a = y3 + 2.0 * a;
  mp_a = y3 + 2.0 * a;
  np_a = y3 + 2.0 * a;
  op_a = y3 + 2.0 * a;
  pp_a = y3 + 2.0 * a;
  qp_a = y3 + 2.0 * a;
  rp_a = y3 + 2.0 * a;
  rp_a = (sqrt((b_y1 * b_y1 + y2 * y2) + rp_a * rp_a) + y3) + 2.0 * a;
  sp_a = y3 + 2.0 * a;
  tp_a = y3 + 2.0 * a;
  up_a = y3 + 2.0 * a;
  up_a = (b_y1 * b_y1 + y2 * y2) + up_a * up_a;
  vp_a = y3 + 2.0 * a;
  vp_a = (sqrt((b_y1 * b_y1 + y2 * y2) + vp_a * vp_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  wp_a = y3 + 2.0 * a;
  xp_a = y3 + 2.0 * a;
  yp_a = y3 + 2.0 * a;
  aq_a = y3 + 2.0 * a;
  bq_a = y3 + 2.0 * a;
  cq_a = y3 + 2.0 * a;
  dq_a = y3 + 2.0 * a;
  eq_a = y3 + 2.0 * a;
  fq_a = y3 + 2.0 * a;
  gq_a = y3 + 2.0 * a;
  hq_a = y3 + 2.0 * a;
  iq_a = y3 + 2.0 * a;
  jq_a = y3 + 2.0 * a;
  kq_a = y3 + 2.0 * a;
  lq_a = y3 + 2.0 * a;
  mq_a = y3 + 2.0 * a;
  nq_a = y3 + 2.0 * a;
  oq_a = y3 + 2.0 * a;
  pq_a = y3 + 2.0 * a;
  qq_a = y3 + 2.0 * a;
  rq_a = y3 + 2.0 * a;
  sq_a = y3 + 2.0 * a;
  tq_a = y3 + 2.0 * a;
  uq_a = y3 + 2.0 * a;
  uq_a = (sqrt((b_y1 * b_y1 + y2 * y2) + uq_a * uq_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  vq_a = y3 + 2.0 * a;
  wq_a = y3 + 2.0 * a;
  xq_a = y3 + 2.0 * a;
  yq_a = y3 + 2.0 * a;
  ar_a = y3 + 2.0 * a;
  br_a = y3 + 2.0 * a;
  cr_a = y3 + 2.0 * a;
  dr_a = y3 + 2.0 * a;
  er_a = y3 + 2.0 * a;
  er_a = (b_y1 * b_y1 + y2 * y2) + er_a * er_a;
  fr_a = y3 + 2.0 * a;
  gr_a = y3 + 2.0 * a;
  hr_a = y3 + 2.0 * a;
  ir_a = y3 + 2.0 * a;
  jr_a = y3 + 2.0 * a;
  kr_a = y3 + 2.0 * a;
  lr_a = y3 + 2.0 * a;
  mr_a = y3 + 2.0 * a;
  nr_a = y3 + 2.0 * a;
  or_a = y3 + 2.0 * a;
  pr_a = y3 + 2.0 * a;
  pr_a = (sqrt((b_y1 * b_y1 + y2 * y2) + pr_a * pr_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  qr_a = y3 + 2.0 * a;
  rr_a = b_y1 * cos(b) - y3 * sin(b);
  sr_a = b_y1 * cos(b) - y3 * sin(b);
  tr_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  k_x = sin(b);
  ur_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  vr_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  wr_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  xr_a = y3 + 2.0 * a;
  yr_a = y3 + 2.0 * a;
  as_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  bs_a = y3 + 2.0 * a;
  l_x = sin(b);
  cs_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  ds_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  es_a = y3 + 2.0 * a;
  fs_a = y3 + 2.0 * a;
  gs_a = y3 + 2.0 * a;
  hs_a = y3 + 2.0 * a;
  is_a = y3 + 2.0 * a;
  js_a = y3 + 2.0 * a;
  ks_a = y3 + 2.0 * a;
  ls_a = y3 + 2.0 * a;
  ls_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ls_a * ls_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  ms_a = y3 + 2.0 * a;
  ns_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  os_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  ps_a = y3 + 2.0 * a;
  qs_a = y3 + 2.0 * a;
  rs_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  ss_a = y3 + 2.0 * a;
  m_x = sin(b);
  ts_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  us_a = y3 + 2.0 * a;
  us_a = (sqrt((b_y1 * b_y1 + y2 * y2) + us_a * us_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  vs_a = y3 + 2.0 * a;
  ws_a = y3 + 2.0 * a;
  xs_a = y3 + 2.0 * a;
  ys_a = y3 + 2.0 * a;
  at_a = y3 + 2.0 * a;
  bt_a = y3 + 2.0 * a;
  ct_a = y3 + 2.0 * a;
  dt_a = y3 + 2.0 * a;
  et_a = y3 + 2.0 * a;
  ft_a = y3 + 2.0 * a;
  gt_a = y3 + 2.0 * a;
  ht_a = y3 + 2.0 * a;
  ht_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ht_a * ht_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  it_a = y3 + 2.0 * a;
  jt_a = y3 + 2.0 * a;
  kt_a = y3 + 2.0 * a;
  lt_a = y3 + 2.0 * a;
  mt_a = y3 + 2.0 * a;
  nt_a = y3 + 2.0 * a;
  ot_a = y3 + 2.0 * a;
  pt_a = y3 + 2.0 * a;
  qt_a = y3 + 2.0 * a;
  rt_a = y3 + 2.0 * a;
  st_a = y3 + 2.0 * a;
  tt_a = y3 + 2.0 * a;
  tt_a = (sqrt((b_y1 * b_y1 + y2 * y2) + tt_a * tt_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  ut_a = y3 + 2.0 * a;
  vt_a = y3 + 2.0 * a;
  wt_a = y3 + 2.0 * a;
  xt_a = y3 + 2.0 * a;
  yt_a = y3 + 2.0 * a;
  au_a = y3 + 2.0 * a;
  au_a = (b_y1 * b_y1 + y2 * y2) + au_a * au_a;
  *e13 = ((((0.5 * B1 * (0.125 * (((2.0 - 2.0 * nu) * (((y2 / (b_a * b_a) * sin
    (b) / (1.0 + y2 * y2 / (c_a * c_a)) + (y2 / sqrt((b_y1 * b_y1 + y2 * y2) +
    y3 * y3) * sin(b) / (b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b))
    * y3 + y2 * sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * (x * x) / (d_a * d_a) *
    b_y1) / (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * (b_x * b_x) /
             (e_a * e_a))) - y2 / (f_a * f_a) * sin(b) / (1.0 + y2 * y2 / (g_a *
    g_a))) + (0.5 * y2 / sqrt((b_y1 * b_y1 + y2 * y2) + h_a * h_a) * sin(b) /
              (b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos
               (b)) * (2.0 * y3 + 4.0 * a) - y2 * sqrt((b_y1 * b_y1 + y2 * y2) +
    i_a * i_a) * (c_x * c_x) / (j_a * j_a) * b_y1) / (1.0 + y2 * y2 * ((b_y1 *
    b_y1 + y2 * y2) + k_a * k_a) * (d_x * d_x) / (l_a * l_a))) - b_y1 * y2 * (((
    -1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / (sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) - y3) * y3 - 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) +
    y3 * y3) / (m_a * m_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * y3
    - 1.0)) - 0.5 / powd_snf((b_y1 * b_y1 + y2 * y2) + n_a * n_a, 1.5) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + o_a * o_a) + y3) + 2.0 * a) * (2.0 * y3 +
    4.0 * a)) - 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + p_a * p_a) / (q_a * q_a) *
    (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + r_a * r_a) * (2.0 * y3 + 4.0 * a) +
     1.0))) - y2 * cos(b) * (((((1.0 / ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin
    (b) * y3 / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 *
                cos(b)) - (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) -
    b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) * y3) - (sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) - b_y1) / sqrt((b_y1 * b_y1 +
    y2 * y2) + y3 * y3) / (s_a * s_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3
    * y3) * y3 - cos(b))) + 0.5 / ((b_y1 * b_y1 + y2 * y2) + t_a * t_a) * sin(b)
    * (2.0 * y3 + 4.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + u_a * u_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b))) - 0.5 * (sqrt((b_y1 * b_y1 + y2 * y2) +
    v_a * v_a) * sin(b) - b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + w_a *
    w_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + x_a * x_a) - b_y1 * sin(b)) +
                 (y3 + 2.0 * a) * cos(b)) * (2.0 * y3 + 4.0 * a)) - (sqrt((b_y1 *
    b_y1 + y2 * y2) + y_a * y_a) * sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2)
    + ab_a * ab_a) / (bb_a * bb_a) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + cb_a *
    cb_a) * (2.0 * y3 + 4.0 * a) + cos(b)))) / M_PI / (1.0 - nu) +
              0.25 * (((((((((((((((-2.0 + 2.0 * nu) * (1.0 - 2.0 * nu) * (-y2 /
    (db_a * db_a) * sin(b) / (1.0 + y2 * y2 / (eb_a * eb_a)) + (0.5 * y2 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + fb_a * fb_a) * sin(b) / (b_y1 * (b_y1 * cos(b) +
    (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b)) * (2.0 * y3 + 4.0 * a) - y2 *
    sqrt((b_y1 * b_y1 + y2 * y2) + gb_a * gb_a) * (e_x * e_x) / (hb_a * hb_a) *
    b_y1) / (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2) + ib_a * ib_a) * (f_x *
    f_x) / (jb_a * jb_a))) * (g_x * g_x) - (1.0 - 2.0 * nu) * y2 / (kb_a * kb_a)
    * (((1.0 - 2.0 * nu) - a / sqrt((b_y1 * b_y1 + y2 * y2) + lb_a * lb_a)) *
    (1.0 / tan(b)) - b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + mb_a * mb_a) + y3)
    + 2.0 * a) * (nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + nb_a * nb_a))) * (0.5 /
    sqrt((b_y1 * b_y1 + y2 * y2) + ob_a * ob_a) * (2.0 * y3 + 4.0 * a) + 1.0)) +
    (1.0 - 2.0 * nu) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + pb_a * pb_a) + y3)
    + 2.0 * a) * ((0.5 * a / powd_snf((b_y1 * b_y1 + y2 * y2) + qb_a * qb_a,
    1.5) * (2.0 * y3 + 4.0 * a) * (1.0 / tan(b)) + b_y1 / (rb_a * rb_a) * (nu +
    a / sqrt((b_y1 * b_y1 + y2 * y2) + sb_a * sb_a)) * (0.5 / sqrt((b_y1 * b_y1
    + y2 * y2) + tb_a * tb_a) * (2.0 * y3 + 4.0 * a) + 1.0)) + 0.5 * b_y1 /
                  ((sqrt((b_y1 * b_y1 + y2 * y2) + ub_a * ub_a) + y3) + 2.0 * a)
                  * a / powd_snf((b_y1 * b_y1 + y2 * y2) + vb_a * vb_a, 1.5) *
                  (2.0 * y3 + 4.0 * a))) - (1.0 - 2.0 * nu) * y2 * cos(b) * (1.0
    / tan(b)) / (wb_a * wb_a) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) +
    xb_a * xb_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + yb_a * yb_a) * (2.0 *
    y3 + 4.0 * a) + cos(b))) - 0.5 * (1.0 - 2.0 * nu) * y2 * cos(b) * (1.0 / tan
    (b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ac_a * ac_a) - b_y1 * sin(b)) + (y3
    + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + bc_a * bc_a,
    1.5) * (2.0 * y3 + 4.0 * a)) + a / powd_snf((b_y1 * b_y1 + y2 * y2) +
    cc_a * cc_a, 1.5) * y2 * (1.0 / tan(b))) - 1.5 * a * y2 * (y3 + a) * (1.0 /
    tan(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + dc_a * dc_a, 2.5) * (2.0 *
    y3 + 4.0 * a)) + y2 / sqrt((b_y1 * b_y1 + y2 * y2) + ec_a * ec_a) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + fc_a * fc_a) + y3) + 2.0 * a) * (((-1.0 + 2.0 *
    nu) * (1.0 / tan(b)) + b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + gc_a * gc_a)
    + y3) + 2.0 * a) * (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + hc_a *
    hc_a))) + a * b_y1 / ((b_y1 * b_y1 + y2 * y2) + ic_a * ic_a))) - 0.5 * y2 *
    (y3 + a) / powd_snf((b_y1 * b_y1 + y2 * y2) + jc_a * jc_a, 1.5) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + kc_a * kc_a) + y3) + 2.0 * a) * (((-1.0 + 2.0 *
    nu) * (1.0 / tan(b)) + b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + lc_a * lc_a)
    + y3) + 2.0 * a) * (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + mc_a *
    mc_a))) + a * b_y1 / ((b_y1 * b_y1 + y2 * y2) + nc_a * nc_a)) * (2.0 * y3 +
    4.0 * a)) - y2 * (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) + oc_a * oc_a) /
    (pc_a * pc_a) * (((-1.0 + 2.0 * nu) * (1.0 / tan(b)) + b_y1 / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + qc_a * qc_a) + y3) + 2.0 * a) * (2.0 * nu + a / sqrt((b_y1
    * b_y1 + y2 * y2) + rc_a * rc_a))) + a * b_y1 / ((b_y1 * b_y1 + y2 * y2) +
    sc_a * sc_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + tc_a * tc_a) * (2.0 *
    y3 + 4.0 * a) + 1.0)) + y2 * (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) + uc_a *
    uc_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + vc_a * vc_a) + y3) + 2.0 * a) *
    ((-b_y1 / (wc_a * wc_a) * (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 * y2) +
    xc_a * xc_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + yc_a * yc_a) * (2.0 *
    y3 + 4.0 * a) + 1.0) - 0.5 * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + ad_a *
    ad_a) + y3) + 2.0 * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + bd_a *
    bd_a, 1.5) * (2.0 * y3 + 4.0 * a)) - a * b_y1 / (cd_a * cd_a) * (2.0 * y3 +
    4.0 * a))) + y2 / sqrt((b_y1 * b_y1 + y2 * y2) + dd_a * dd_a) / ((sqrt((b_y1
    * b_y1 + y2 * y2) + ed_a * ed_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))
    * (cos(b) / ((sqrt((b_y1 * b_y1 + y2 * y2) + fd_a * fd_a) - b_y1 * sin(b)) +
                 (y3 + 2.0 * a) * cos(b)) * (((sqrt((b_y1 * b_y1 + y2 * y2) +
    gd_a * gd_a) * cos(b) + y3) + 2.0 * a) * ((1.0 - 2.0 * nu) * cos(b) - a /
    sqrt((b_y1 * b_y1 + y2 * y2) + hd_a * hd_a)) * (1.0 / tan(b)) + (2.0 - 2.0 *
    nu) * (sqrt((b_y1 * b_y1 + y2 * y2) + id_a * id_a) * sin(b) - b_y1) * cos(b))
       - a * (y3 + 2.0 * a) * cos(b) * (1.0 / tan(b)) / ((b_y1 * b_y1 + y2 * y2)
    + jd_a * jd_a))) - 0.5 * y2 * (y3 + a) / powd_snf((b_y1 * b_y1 + y2 * y2)
    + kd_a * kd_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ld_a * ld_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + md_a * md_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((sqrt((b_y1 *
    b_y1 + y2 * y2) + nd_a * nd_a) * cos(b) + y3) + 2.0 * a) * ((1.0 - 2.0 * nu)
    * cos(b) - a / sqrt((b_y1 * b_y1 + y2 * y2) + od_a * od_a)) * (1.0 / tan(b))
    + (2.0 - 2.0 * nu) * (sqrt((b_y1 * b_y1 + y2 * y2) + pd_a * pd_a) * sin(b) -
    b_y1) * cos(b)) - a * (y3 + 2.0 * a) * cos(b) * (1.0 / tan(b)) / ((b_y1 *
    b_y1 + y2 * y2) + qd_a * qd_a)) * (2.0 * y3 + 4.0 * a)) - y2 * (y3 + a) /
                       sqrt((b_y1 * b_y1 + y2 * y2) + rd_a * rd_a) / (sd_a *
    sd_a) * (cos(b) / ((sqrt((b_y1 * b_y1 + y2 * y2) + td_a * td_a) - b_y1 * sin
                        (b)) + (y3 + 2.0 * a) * cos(b)) * (((sqrt((b_y1 * b_y1 +
    y2 * y2) + ud_a * ud_a) * cos(b) + y3) + 2.0 * a) * ((1.0 - 2.0 * nu) * cos
    (b) - a / sqrt((b_y1 * b_y1 + y2 * y2) + vd_a * vd_a)) * (1.0 / tan(b)) +
    (2.0 - 2.0 * nu) * (sqrt((b_y1 * b_y1 + y2 * y2) + wd_a * wd_a) * sin(b) -
                        b_y1) * cos(b)) - a * (y3 + 2.0 * a) * cos(b) * (1.0 /
    tan(b)) / ((b_y1 * b_y1 + y2 * y2) + xd_a * xd_a)) * (0.5 / sqrt((b_y1 *
    b_y1 + y2 * y2) + yd_a * yd_a) * (2.0 * y3 + 4.0 * a) + cos(b))) + y2 * (y3
    + a) / sqrt((b_y1 * b_y1 + y2 * y2) + ae_a * ae_a) / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + be_a * be_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
                      (((-cos(b) / (ce_a * ce_a) * (((sqrt((b_y1 * b_y1 + y2 *
    y2) + de_a * de_a) * cos(b) + y3) + 2.0 * a) * ((1.0 - 2.0 * nu) * cos(b) -
    a / sqrt((b_y1 * b_y1 + y2 * y2) + ee_a * ee_a)) * (1.0 / tan(b)) + (2.0 -
    2.0 * nu) * (sqrt((b_y1 * b_y1 + y2 * y2) + fe_a * fe_a) * sin(b) - b_y1) *
    cos(b)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + ge_a * ge_a) * (2.0 * y3 +
    4.0 * a) + cos(b)) + cos(b) / ((sqrt((b_y1 * b_y1 + y2 * y2) + he_a * he_a)
    - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((0.5 / sqrt((b_y1 * b_y1 +
    y2 * y2) + ie_a * ie_a) * cos(b) * (2.0 * y3 + 4.0 * a) + 1.0) * ((1.0 - 2.0
    * nu) * cos(b) - a / sqrt((b_y1 * b_y1 + y2 * y2) + je_a * je_a)) * (1.0 /
    tan(b)) + 0.5 * ((sqrt((b_y1 * b_y1 + y2 * y2) + ke_a * ke_a) * cos(b) + y3)
                     + 2.0 * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + le_a
    * le_a, 1.5) * (2.0 * y3 + 4.0 * a) * (1.0 / tan(b))) + 0.5 * (2.0 - 2.0 *
    nu) / sqrt((b_y1 * b_y1 + y2 * y2) + me_a * me_a) * sin(b) * (2.0 * y3 + 4.0
    * a) * cos(b))) - a * cos(b) * (1.0 / tan(b)) / ((b_y1 * b_y1 + y2 * y2) +
    ne_a * ne_a)) + a * (y3 + 2.0 * a) * cos(b) * (1.0 / tan(b)) / (oe_a * oe_a)
                       * (2.0 * y3 + 4.0 * a))) / M_PI / (1.0 - nu))
             + 0.5 * B2 * (0.125 * ((((((((((-1.0 + 2.0 * nu) * (((1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * y3 - 1.0) / (sqrt((b_y1 * b_y1 + y2 *
    y2) + y3 * y3) - y3) + (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + pe_a * pe_a) *
    (2.0 * y3 + 4.0 * a) + 1.0) / ((sqrt((b_y1 * b_y1 + y2 * y2) + qe_a * qe_a)
    + y3) + 2.0 * a)) - cos(b) * ((1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3)
    * y3 - cos(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b))
                      - y3 * cos(b)) + (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) +
    re_a * re_a) * (2.0 * y3 + 4.0 * a) + cos(b)) / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + se_a * se_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)))) + b_y1 *
    b_y1 * (((-1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / (sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3) * y3 - 1.0 / sqrt((b_y1 * b_y1 +
    y2 * y2) + y3 * y3) / (te_a * te_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) +
    y3 * y3) * y3 - 1.0)) - 0.5 / powd_snf((b_y1 * b_y1 + y2 * y2) + ue_a *
    ue_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ve_a * ve_a) + y3) + 2.0 * a) *
             (2.0 * y3 + 4.0 * a)) - 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + we_a *
    we_a) / (xe_a * xe_a) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + ye_a * ye_a) *
    (2.0 * y3 + 4.0 * a) + 1.0))) - sin(b) * (sqrt((b_y1 * b_y1 + y2 * y2) + y3 *
    y3) * sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b))) + (b_y1
    * cos(b) - y3 * sin(b)) / ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) * y3 /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b))) -
    (b_y1 * cos(b) - y3 * sin(b)) * (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) *
    sin(b) - b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) *
    y3) - (b_y1 * cos(b) - y3 * sin(b)) * (sqrt((b_y1 * b_y1 + y2 * y2) + y3 *
    y3) * sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (af_a *
    af_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * y3 - cos(b))) + sin
    (b) * (sqrt((b_y1 * b_y1 + y2 * y2) + bf_a * bf_a) * sin(b) - b_y1) / sqrt
    ((b_y1 * b_y1 + y2 * y2) + cf_a * cf_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    df_a * df_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) + 0.5 * (b_y1 *
    cos(b) + (y3 + 2.0 * a) * sin(b)) / ((b_y1 * b_y1 + y2 * y2) + ef_a * ef_a) *
    sin(b) * (2.0 * y3 + 4.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ff_a * ff_a)
    - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) - 0.5 * (b_y1 * cos(b) + (y3 +
    2.0 * a) * sin(b)) * (sqrt((b_y1 * b_y1 + y2 * y2) + gf_a * gf_a) * sin(b) -
    b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + hf_a * hf_a, 1.5) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + if_a * if_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b)) * (2.0 * y3 + 4.0 * a)) - (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) *
    (sqrt((b_y1 * b_y1 + y2 * y2) + jf_a * jf_a) * sin(b) - b_y1) / sqrt((b_y1 *
    b_y1 + y2 * y2) + kf_a * kf_a) / (lf_a * lf_a) * (0.5 / sqrt((b_y1 * b_y1 +
    y2 * y2) + mf_a * mf_a) * (2.0 * y3 + 4.0 * a) + cos(b))) /
              M_PI / (1.0 - nu) + 0.25 * (((((((((((((1.0 - 2.0 *
    nu) * (((2.0 - 2.0 * nu) * (h_x * h_x) + nu) * (0.5 / sqrt((b_y1 * b_y1 + y2
    * y2) + nf_a * nf_a) * (2.0 * y3 + 4.0 * a) + 1.0) / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + of_a * of_a) + y3) + 2.0 * a) - ((2.0 - 2.0 * nu) * (i_x * i_x) +
    1.0) * cos(b) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + pf_a * pf_a) * (2.0 *
    y3 + 4.0 * a) + cos(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + qf_a * qf_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) - (1.0 - 2.0 * nu) / (rf_a * rf_a)
    * (((((-1.0 + 2.0 * nu) * b_y1 * (1.0 / tan(b)) + nu * (y3 + 2.0 * a)) - a)
        + a * b_y1 * (1.0 / tan(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + sf_a * sf_a))
       + b_y1 * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + tf_a * tf_a) + y3) + 2.0
                        * a) * (nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + uf_a *
    uf_a))) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + vf_a * vf_a) * (2.0 * y3 +
    4.0 * a) + 1.0)) + (1.0 - 2.0 * nu) / ((sqrt((b_y1 * b_y1 + y2 * y2) + wf_a *
    wf_a) + y3) + 2.0 * a) * (((nu - 0.5 * a * b_y1 * (1.0 / tan(b)) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + xf_a * xf_a, 1.5) * (2.0 * y3 + 4.0 *
    a)) - b_y1 * b_y1 / (yf_a * yf_a) * (nu + a / sqrt((b_y1 * b_y1 + y2 * y2) +
    ag_a * ag_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + bg_a * bg_a) * (2.0 *
    y3 + 4.0 * a) + 1.0)) - 0.5 * (b_y1 * b_y1) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + cg_a * cg_a) + y3) + 2.0 * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2) +
    dg_a * dg_a, 1.5) * (2.0 * y3 + 4.0 * a))) + (1.0 - 2.0 * nu) * (1.0 / tan(b))
    / (eg_a * eg_a) * ((b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * cos(b) - a *
                       (sqrt((b_y1 * b_y1 + y2 * y2) + fg_a * fg_a) * sin(b) -
                        b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + gg_a * gg_a) /
                       cos(b)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + hg_a *
    hg_a) * (2.0 * y3 + 4.0 * a) + cos(b))) - (1.0 - 2.0 * nu) * (1.0 / tan(b)) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + ig_a * ig_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)) * ((cos(b) * sin(b) - 0.5 * a / ((b_y1 * b_y1 + y2 * y2) + jg_a
    * jg_a) * sin(b) * (2.0 * y3 + 4.0 * a) / cos(b)) + 0.5 * a * (sqrt((b_y1 *
    b_y1 + y2 * y2) + kg_a * kg_a) * sin(b) - b_y1) / powd_snf((b_y1 * b_y1 +
    y2 * y2) + lg_a * lg_a, 1.5) / cos(b) * (2.0 * y3 + 4.0 * a))) - a /
    powd_snf((b_y1 * b_y1 + y2 * y2) + mg_a * mg_a, 1.5) * b_y1 * (1.0 / tan
    (b))) + 1.5 * a * b_y1 * (y3 + a) * (1.0 / tan(b)) / powd_snf((b_y1 *
    b_y1 + y2 * y2) + ng_a * ng_a, 2.5) * (2.0 * y3 + 4.0 * a)) + 1.0 / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + og_a * og_a) + y3) + 2.0 * a) * (((2.0 * nu + 1.0
    / sqrt((b_y1 * b_y1 + y2 * y2) + pg_a * pg_a) * ((1.0 - 2.0 * nu) * b_y1 *
    (1.0 / tan(b)) + a)) - b_y1 * b_y1 / sqrt((b_y1 * b_y1 + y2 * y2) + qg_a *
    qg_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + rg_a * rg_a) + y3) + 2.0 * a) *
    (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + sg_a * sg_a))) - a * (b_y1 *
    b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + tg_a * tg_a, 1.5))) - (y3 + a)
    / (ug_a * ug_a) * (((2.0 * nu + 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + vg_a *
    vg_a) * ((1.0 - 2.0 * nu) * b_y1 * (1.0 / tan(b)) + a)) - b_y1 * b_y1 / sqrt
                        ((b_y1 * b_y1 + y2 * y2) + wg_a * wg_a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + xg_a * xg_a) + y3) + 2.0 * a) * (2.0 * nu + a / sqrt((b_y1
    * b_y1 + y2 * y2) + yg_a * yg_a))) - a * (b_y1 * b_y1) / powd_snf((b_y1 *
    b_y1 + y2 * y2) + ah_a * ah_a, 1.5)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) +
    bh_a * bh_a) * (2.0 * y3 + 4.0 * a) + 1.0)) + (y3 + a) / ((sqrt((b_y1 * b_y1
    + y2 * y2) + ch_a * ch_a) + y3) + 2.0 * a) * ((((-0.5 / powd_snf((b_y1 *
    b_y1 + y2 * y2) + dh_a * dh_a, 1.5) * ((1.0 - 2.0 * nu) * b_y1 * (1.0 / tan
    (b)) + a) * (2.0 * y3 + 4.0 * a) + 0.5 * (b_y1 * b_y1) / powd_snf((b_y1 *
    b_y1 + y2 * y2) + eh_a * eh_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + fh_a *
    fh_a) + y3) + 2.0 * a) * (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + gh_a
    * gh_a)) * (2.0 * y3 + 4.0 * a)) + b_y1 * b_y1 / sqrt((b_y1 * b_y1 + y2 * y2)
    + hh_a * hh_a) / (ih_a * ih_a) * (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 * y2)
    + jh_a * jh_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + kh_a * kh_a) * (2.0 *
    y3 + 4.0 * a) + 1.0)) + 0.5 * (b_y1 * b_y1) / (lh_a * lh_a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + mh_a * mh_a) + y3) + 2.0 * a) * a * (2.0 * y3 + 4.0 * a))
    + 1.5 * a * (b_y1 * b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + nh_a *
    nh_a, 2.5) * (2.0 * y3 + 4.0 * a))) + 1.0 / tan(b) / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + oh_a * oh_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((-cos
    (b) * sin(b) + a * b_y1 * (y3 + 2.0 * a) / powd_snf((b_y1 * b_y1 + y2 *
    y2) + ph_a * ph_a, 1.5) / cos(b)) + (sqrt((b_y1 * b_y1 + y2 * y2) + qh_a *
    qh_a) * sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + rh_a * rh_a) * ((2.0
    - 2.0 * nu) * cos(b) - ((sqrt((b_y1 * b_y1 + y2 * y2) + sh_a * sh_a) * cos(b)
    + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + th_a * th_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2)
    + uh_a * uh_a) / cos(b))))) - (y3 + a) * (1.0 / tan(b)) / (vh_a * vh_a) * ((
    -cos(b) * sin(b) + a * b_y1 * (y3 + 2.0 * a) / powd_snf((b_y1 * b_y1 + y2
    * y2) + wh_a * wh_a, 1.5) / cos(b)) + (sqrt((b_y1 * b_y1 + y2 * y2) + xh_a *
    xh_a) * sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + yh_a * yh_a) * ((2.0
    - 2.0 * nu) * cos(b) - ((sqrt((b_y1 * b_y1 + y2 * y2) + ai_a * ai_a) * cos(b)
    + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + bi_a * bi_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2)
    + ci_a * ci_a) / cos(b)))) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + di_a *
    di_a) * (2.0 * y3 + 4.0 * a) + cos(b))) + (y3 + a) * (1.0 / tan(b)) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + ei_a * ei_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b)) * ((((a / powd_snf((b_y1 * b_y1 + y2 * y2) + fi_a * fi_a, 1.5) /
                  cos(b) * b_y1 - 1.5 * a * b_y1 * (y3 + 2.0 * a) / powd_snf
                  ((b_y1 * b_y1 + y2 * y2) + gi_a * gi_a, 2.5) / cos(b) * (2.0 *
    y3 + 4.0 * a)) + 0.5 / ((b_y1 * b_y1 + y2 * y2) + hi_a * hi_a) * sin(b) *
                 (2.0 * y3 + 4.0 * a) * ((2.0 - 2.0 * nu) * cos(b) - ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + ii_a * ii_a) * cos(b) + y3) + 2.0 * a) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + ji_a * ji_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b)) * (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2) + ki_a * ki_a) / cos(b))))
                - 0.5 * (sqrt((b_y1 * b_y1 + y2 * y2) + li_a * li_a) * sin(b) -
    b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + mi_a * mi_a, 1.5) * ((2.0 -
    2.0 * nu) * cos(b) - ((sqrt((b_y1 * b_y1 + y2 * y2) + ni_a * ni_a) * cos(b)
    + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + oi_a * oi_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2)
    + pi_a * pi_a) / cos(b))) * (2.0 * y3 + 4.0 * a)) + (sqrt((b_y1 * b_y1 + y2 *
    y2) + qi_a * qi_a) * sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + ri_a *
    ri_a) * ((-(0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + si_a * si_a) * cos(b) *
                (2.0 * y3 + 4.0 * a) + 1.0) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    ti_a * ti_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt
    ((b_y1 * b_y1 + y2 * y2) + ui_a * ui_a) / cos(b)) + ((sqrt((b_y1 * b_y1 + y2
    * y2) + vi_a * vi_a) * cos(b) + y3) + 2.0 * a) / (wi_a * wi_a) * (1.0 + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + xi_a * xi_a) / cos(b)) * (0.5 / sqrt((b_y1 *
    b_y1 + y2 * y2) + yi_a * yi_a) * (2.0 * y3 + 4.0 * a) + cos(b))) + 0.5 *
             ((sqrt((b_y1 * b_y1 + y2 * y2) + aj_a * aj_a) * cos(b) + y3) + 2.0 *
              a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + bj_a * bj_a) - b_y1 * sin(b))
                    + (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 * b_y1 +
    y2 * y2) + cj_a * cj_a, 1.5) / cos(b) * (2.0 * y3 + 4.0 * a)))) /
              M_PI / (1.0 - nu))) + 0.5 * B3 * (0.125 * y2 * sin(b)
             * (((((1.0 / ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) * y3 /
                    ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) -
                     y3 * cos(b)) - (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) *
    sin(b) - b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) /
                    ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) -
                     y3 * cos(b)) * y3) - (sqrt((b_y1 * b_y1 + y2 * y2) + y3 *
    y3) * sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (dj_a *
    dj_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * y3 - cos(b))) + 0.5
                  / ((b_y1 * b_y1 + y2 * y2) + ej_a * ej_a) * sin(b) * (2.0 * y3
    + 4.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + fj_a * fj_a) - b_y1 * sin(b))
                  + (y3 + 2.0 * a) * cos(b))) - 0.5 * (sqrt((b_y1 * b_y1 + y2 *
    y2) + gj_a * gj_a) * sin(b) - b_y1) / powd_snf((b_y1 * b_y1 + y2 * y2) +
    hj_a * hj_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ij_a * ij_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * (2.0 * y3 + 4.0 * a)) - (sqrt((b_y1 *
    b_y1 + y2 * y2) + jj_a * jj_a) * sin(b) - b_y1) / sqrt((b_y1 * b_y1 + y2 *
    y2) + kj_a * kj_a) / (lj_a * lj_a) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) +
    mj_a * mj_a) * (2.0 * y3 + 4.0 * a) + cos(b))) / M_PI / (1.0 -
              nu) + 0.25 * ((((((((1.0 - 2.0 * nu) * (((-y2 / (nj_a * nj_a) *
    (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2) + oj_a * oj_a)) * (0.5 / sqrt((b_y1 *
    b_y1 + y2 * y2) + pj_a * pj_a) * (2.0 * y3 + 4.0 * a) + 1.0) - 0.5 * y2 /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + qj_a * qj_a) + y3) + 2.0 * a) * a /
    powd_snf((b_y1 * b_y1 + y2 * y2) + rj_a * rj_a, 1.5) * (2.0 * y3 + 4.0 *
    a)) + y2 * cos(b) / (sj_a * sj_a) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 *
    y2) + tj_a * tj_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + uj_a * uj_a) *
    (2.0 * y3 + 4.0 * a) + cos(b))) + 0.5 * y2 * cos(b) / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + vj_a * vj_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * a /
    powd_snf((b_y1 * b_y1 + y2 * y2) + wj_a * wj_a, 1.5) * (2.0 * y3 + 4.0 *
    a)) - y2 / sqrt((b_y1 * b_y1 + y2 * y2) + xj_a * xj_a) * (a / ((b_y1 * b_y1
    + y2 * y2) + yj_a * yj_a) + 1.0 / ((sqrt((b_y1 * b_y1 + y2 * y2) + ak_a *
    ak_a) + y3) + 2.0 * a))) + 0.5 * y2 * (y3 + a) / powd_snf((b_y1 * b_y1 +
    y2 * y2) + bk_a * bk_a, 1.5) * (a / ((b_y1 * b_y1 + y2 * y2) + ck_a * ck_a)
    + 1.0 / ((sqrt((b_y1 * b_y1 + y2 * y2) + dk_a * dk_a) + y3) + 2.0 * a)) *
    (2.0 * y3 + 4.0 * a)) - y2 * (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) + ek_a *
    ek_a) * (-a / (fk_a * fk_a) * (2.0 * y3 + 4.0 * a) - 1.0 / (gk_a * gk_a) *
             (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + hk_a * hk_a) * (2.0 * y3 +
    4.0 * a) + 1.0))) + y2 * cos(b) / sqrt((b_y1 * b_y1 + y2 * y2) + ik_a * ik_a)
    / ((sqrt((b_y1 * b_y1 + y2 * y2) + jk_a * jk_a) - b_y1 * sin(b)) + (y3 + 2.0
    * a) * cos(b)) * (((sqrt((b_y1 * b_y1 + y2 * y2) + kk_a * kk_a) * cos(b) +
                        y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + lk_a *
    lk_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1
    * b_y1 + y2 * y2) + mk_a * mk_a)) + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 *
    y2) + nk_a * nk_a))) - 0.5 * y2 * (y3 + a) * cos(b) / powd_snf((b_y1 *
    b_y1 + y2 * y2) + ok_a * ok_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + pk_a *
    pk_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((sqrt((b_y1 * b_y1 +
    y2 * y2) + qk_a * qk_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2
    * y2) + rk_a * rk_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) +
    a / sqrt((b_y1 * b_y1 + y2 * y2) + sk_a * sk_a)) + a * (y3 + 2.0 * a) /
    ((b_y1 * b_y1 + y2 * y2) + tk_a * tk_a)) * (2.0 * y3 + 4.0 * a)) - y2 * (y3
    + a) * cos(b) / sqrt((b_y1 * b_y1 + y2 * y2) + uk_a * uk_a) / (vk_a * vk_a) *
    (((sqrt((b_y1 * b_y1 + y2 * y2) + wk_a * wk_a) * cos(b) + y3) + 2.0 * a) /
     ((sqrt((b_y1 * b_y1 + y2 * y2) + xk_a * xk_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + yk_a * yk_a)) +
     a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) + al_a * al_a)) * (0.5 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + bl_a * bl_a) * (2.0 * y3 + 4.0 * a) + cos(b))) +
              y2 * (y3 + a) * cos(b) / sqrt((b_y1 * b_y1 + y2 * y2) + cl_a *
    cl_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + dl_a * dl_a) - b_y1 * sin(b)) + (y3
    + 2.0 * a) * cos(b)) * (((((0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + el_a * el_a)
    * cos(b) * (2.0 * y3 + 4.0 * a) + 1.0) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    fl_a * fl_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + gl_a * gl_a)) - ((sqrt((b_y1 * b_y1 + y2 * y2)
    + hl_a * hl_a) * cos(b) + y3) + 2.0 * a) / (il_a * il_a) * (cos(b) + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + jl_a * jl_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2
    * y2) + kl_a * kl_a) * (2.0 * y3 + 4.0 * a) + cos(b))) - 0.5 * ((sqrt((b_y1 *
    b_y1 + y2 * y2) + ll_a * ll_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + ml_a * ml_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    a / powd_snf((b_y1 * b_y1 + y2 * y2) + nl_a * nl_a, 1.5) * (2.0 * y3 +
    4.0 * a)) + a / ((b_y1 * b_y1 + y2 * y2) + ol_a * ol_a)) - a * (y3 + 2.0 * a)
    / (pl_a * pl_a) * (2.0 * y3 + 4.0 * a))) / M_PI / (1.0 - nu)))
           + 0.5 * B1 * (0.125 * y2 * ((-1.0 / powd_snf((b_y1 * b_y1 + y2 *
    y2) + y3 * y3, 1.5) * b_y1 + 1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) +
    ql_a * ql_a, 1.5) * b_y1) - cos(b) * (((((1.0 / ((b_y1 * b_y1 + y2 * y2) +
    y3 * y3) * cos(b) * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 *
    sin(b)) - y3 * cos(b)) - (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b) -
    y3) / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) * b_y1) - (sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b) - y3) / sqrt((b_y1 * b_y1 + y2 *
    y2) + y3 * y3) / (rl_a * rl_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 *
    y3) * b_y1 - sin(b))) - 1.0 / ((b_y1 * b_y1 + y2 * y2) + sl_a * sl_a) * cos
    (b) * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + tl_a * tl_a) - b_y1 * sin(b))
                  + (y3 + 2.0 * a) * cos(b))) + ((sqrt((b_y1 * b_y1 + y2 * y2) +
    ul_a * ul_a) * cos(b) + y3) + 2.0 * a) / powd_snf((b_y1 * b_y1 + y2 * y2)
    + vl_a * vl_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + wl_a * wl_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * b_y1) + ((sqrt((b_y1 * b_y1 + y2 * y2)
    + xl_a * xl_a) * cos(b) + y3) + 2.0 * a) / sqrt((b_y1 * b_y1 + y2 * y2) +
    yl_a * yl_a) / (am_a * am_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + bm_a *
    bm_a) * b_y1 - sin(b)))) / M_PI / (1.0 - nu) + 0.25 *
            ((((((2.0 - 2.0 * nu) * (((((1.0 - 2.0 * nu) * ((y2 / (b_y1 * b_y1) /
    (1.0 + y2 * y2 / (b_y1 * b_y1)) - y2 / (cm_a * cm_a) * cos(b) / (1.0 + y2 *
    y2 / (dm_a * dm_a))) + (y2 / sqrt((b_y1 * b_y1 + y2 * y2) + em_a * em_a) *
    sin(b) / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b))
    * b_y1 - y2 * sqrt((b_y1 * b_y1 + y2 * y2) + fm_a * fm_a) * sin(b) / (gm_a *
    gm_a) * (2.0 * b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b))) / (1.0 + y2 * y2 *
    ((b_y1 * b_y1 + y2 * y2) + hm_a * hm_a) * (j_x * j_x) / (im_a * im_a))) *
    (1.0 / tan(b)) - b_y1 / (jm_a * jm_a) * (2.0 * nu + a / sqrt((b_y1 * b_y1 +
    y2 * y2) + km_a * km_a)) / sqrt((b_y1 * b_y1 + y2 * y2) + lm_a * lm_a) * y2)
    - y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + mm_a * mm_a) + y3) + 2.0 * a) * a /
    powd_snf((b_y1 * b_y1 + y2 * y2) + nm_a * nm_a, 1.5) * b_y1) + y2 * cos(b)
    / (om_a * om_a) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + pm_a * pm_a))
    * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + qm_a * qm_a) * b_y1 - sin(b))) + y2 *
    cos(b) / ((sqrt((b_y1 * b_y1 + y2 * y2) + rm_a * rm_a) - b_y1 * sin(b)) +
              (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 * y2)
    + sm_a * sm_a, 1.5) * b_y1) - y2 * (y3 + a) / powd_snf((b_y1 * b_y1 + y2 *
    y2) + tm_a * tm_a, 1.5) * (2.0 * nu / ((sqrt((b_y1 * b_y1 + y2 * y2) + um_a *
    um_a) + y3) + 2.0 * a) + a / ((b_y1 * b_y1 + y2 * y2) + vm_a * vm_a)) * b_y1)
                + y2 * (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) + wm_a * wm_a) *
                (-2.0 * nu / (xm_a * xm_a) / sqrt((b_y1 * b_y1 + y2 * y2) + ym_a
    * ym_a) * b_y1 - 2.0 * a / (an_a * an_a) * b_y1)) - y2 * (y3 + a) * cos(b) /
               powd_snf((b_y1 * b_y1 + y2 * y2) + bn_a * bn_a, 1.5) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + cn_a * cn_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b)) * (((1.0 - 2.0 * nu) - ((sqrt((b_y1 * b_y1 + y2 * y2) + dn_a * dn_a)
    * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + en_a * en_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1 * b_y1
    + y2 * y2) + fn_a * fn_a))) - a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2)
    + gn_a * gn_a)) * b_y1) - y2 * (y3 + a) * cos(b) / sqrt((b_y1 * b_y1 + y2 *
    y2) + hn_a * hn_a) / (in_a * in_a) * (((1.0 - 2.0 * nu) - ((sqrt((b_y1 *
    b_y1 + y2 * y2) + jn_a * jn_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + kn_a * kn_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + ln_a * ln_a))) - a * (y3 + 2.0 *
    a) / ((b_y1 * b_y1 + y2 * y2) + mn_a * mn_a)) * (1.0 / sqrt((b_y1 * b_y1 +
    y2 * y2) + nn_a * nn_a) * b_y1 - sin(b))) + y2 * (y3 + a) * cos(b) / sqrt
             ((b_y1 * b_y1 + y2 * y2) + on_a * on_a) / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + pn_a * pn_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((-1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + qn_a * qn_a) * cos(b) * b_y1 / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + rn_a * rn_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + sn_a * sn_a)) + ((sqrt((b_y1 *
    b_y1 + y2 * y2) + tn_a * tn_a) * cos(b) + y3) + 2.0 * a) / (un_a * un_a) *
    (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + vn_a * vn_a)) * (1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + wn_a * wn_a) * b_y1 - sin(b))) + ((sqrt((b_y1 *
    b_y1 + y2 * y2) + xn_a * xn_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + yn_a * yn_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    a / powd_snf((b_y1 * b_y1 + y2 * y2) + ao_a * ao_a, 1.5) * b_y1) + 2.0 *
              a * (y3 + 2.0 * a) / (bo_a * bo_a) * b_y1)) / M_PI /
            (1.0 - nu))) + 0.5 * B2 * (0.125 * ((((((((((((-1.0 + 2.0 * nu) *
    sin(b) * ((1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * b_y1 - sin(b)) /
              ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 *
               cos(b)) - (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + co_a * co_a) *
    b_y1 - sin(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + do_a * do_a) - b_y1 * sin
                       (b)) + (y3 + 2.0 * a) * cos(b))) - 1.0 / sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3)) + 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + eo_a *
    eo_a)) - b_y1 * (-1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) *
                     b_y1 + 1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + fo_a *
    fo_a, 1.5) * b_y1)) + cos(b) * (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) *
    cos(b) - y3) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / ((sqrt((b_y1 * b_y1
    + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b))) + (b_y1 * cos(b) - y3
    * sin(b)) / ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b) * b_y1 / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b))) - (b_y1
    * cos(b) - y3 * sin(b)) * (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b)
    - y3) / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) * b_y1) - (b_y1 *
    cos(b) - y3 * sin(b)) * (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b) -
    y3) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (go_a * go_a) * (1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * b_y1 - sin(b))) - cos(b) * ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + ho_a * ho_a) * cos(b) + y3) + 2.0 * a) / sqrt
    ((b_y1 * b_y1 + y2 * y2) + io_a * io_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    jo_a * jo_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) - (b_y1 * cos(b) +
    (y3 + 2.0 * a) * sin(b)) / ((b_y1 * b_y1 + y2 * y2) + ko_a * ko_a) * cos(b) *
              b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + lo_a * lo_a) - b_y1 * sin
                       (b)) + (y3 + 2.0 * a) * cos(b))) + (b_y1 * cos(b) + (y3 +
    2.0 * a) * sin(b)) * ((sqrt((b_y1 * b_y1 + y2 * y2) + mo_a * mo_a) * cos(b)
    + y3) + 2.0 * a) / powd_snf((b_y1 * b_y1 + y2 * y2) + no_a * no_a, 1.5) /
             ((sqrt((b_y1 * b_y1 + y2 * y2) + oo_a * oo_a) - b_y1 * sin(b)) +
              (y3 + 2.0 * a) * cos(b)) * b_y1) + (b_y1 * cos(b) + (y3 + 2.0 * a)
             * sin(b)) * ((sqrt((b_y1 * b_y1 + y2 * y2) + po_a * po_a) * cos(b)
              + y3) + 2.0 * a) / sqrt((b_y1 * b_y1 + y2 * y2) + qo_a * qo_a) /
            (ro_a * ro_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + so_a * so_a) *
             b_y1 - sin(b))) / M_PI / (1.0 - nu) + 0.25 *
           (((((((((((-2.0 + 2.0 * nu) * (1.0 - 2.0 * nu) * (1.0 / tan(b)) *
                     (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + to_a * to_a) * b_y1 /
                      ((sqrt((b_y1 * b_y1 + y2 * y2) + uo_a * uo_a) + y3) + 2.0 *
                       a) - cos(b) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + vo_a *
    vo_a) * b_y1 - sin(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + wo_a * wo_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) - (2.0 - 2.0 * nu) / ((sqrt((b_y1
    * b_y1 + y2 * y2) + xo_a * xo_a) + y3) + 2.0 * a) * (2.0 * nu + a / sqrt
    ((b_y1 * b_y1 + y2 * y2) + yo_a * yo_a))) + (2.0 - 2.0 * nu) * (b_y1 * b_y1)
                    / (ap_a * ap_a) * (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 *
    y2) + bp_a * bp_a)) / sqrt((b_y1 * b_y1 + y2 * y2) + cp_a * cp_a)) + (2.0 -
    2.0 * nu) * (b_y1 * b_y1) / ((sqrt((b_y1 * b_y1 + y2 * y2) + dp_a * dp_a) +
    y3) + 2.0 * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + ep_a * ep_a, 1.5))
                  + (2.0 - 2.0 * nu) * cos(b) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    fp_a * fp_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + gp_a * gp_a))) - (2.0 - 2.0 * nu) * (b_y1 *
    cos(b) + (y3 + 2.0 * a) * sin(b)) / (hp_a * hp_a) * (cos(b) + a / sqrt((b_y1
    * b_y1 + y2 * y2) + ip_a * ip_a)) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) +
    jp_a * jp_a) * b_y1 - sin(b))) - (2.0 - 2.0 * nu) * (b_y1 * cos(b) + (y3 +
    2.0 * a) * sin(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + kp_a * kp_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 * y2)
    + lp_a * lp_a, 1.5) * b_y1) - (y3 + a) / powd_snf((b_y1 * b_y1 + y2 * y2)
    + mp_a * mp_a, 1.5) * (((1.0 - 2.0 * nu) * (1.0 / tan(b)) - 2.0 * nu * b_y1 /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + np_a * np_a) + y3) + 2.0 * a)) - a * b_y1 /
    ((b_y1 * b_y1 + y2 * y2) + op_a * op_a)) * b_y1) + (y3 + a) / sqrt((b_y1 *
    b_y1 + y2 * y2) + pp_a * pp_a) * (((-2.0 * nu / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + qp_a * qp_a) + y3) + 2.0 * a) + 2.0 * nu * (b_y1 * b_y1) / (rp_a *
    rp_a) / sqrt((b_y1 * b_y1 + y2 * y2) + sp_a * sp_a)) - a / ((b_y1 * b_y1 +
    y2 * y2) + tp_a * tp_a)) + 2.0 * a * (b_y1 * b_y1) / (up_a * up_a))) + (y3 +
              a) / (vp_a * vp_a) * ((cos(b) * sin(b) + ((sqrt((b_y1 * b_y1 + y2 *
    y2) + wp_a * wp_a) * cos(b) + y3) + 2.0 * a) * (1.0 / tan(b)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + xp_a * xp_a) * ((2.0 - 2.0 * nu) * cos(b) - ((sqrt((b_y1 *
    b_y1 + y2 * y2) + yp_a * yp_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + aq_a * aq_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))))
              + a / sqrt((b_y1 * b_y1 + y2 * y2) + bq_a * bq_a) * ((sin(b) - (y3
    + 2.0 * a) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / ((b_y1 * b_y1 + y2 *
    y2) + cq_a * cq_a)) - (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + dq_a * dq_a) * cos(b) + y3) + 2.0 * a) / sqrt
    ((b_y1 * b_y1 + y2 * y2) + eq_a * eq_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    fq_a * fq_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)))) * (1.0 / sqrt
              ((b_y1 * b_y1 + y2 * y2) + gq_a * gq_a) * b_y1 - sin(b))) - (y3 +
             a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + hq_a * hq_a) - b_y1 * sin(b))
                   + (y3 + 2.0 * a) * cos(b)) * ((((1.0 / ((b_y1 * b_y1 + y2 *
    y2) + iq_a * iq_a) * cos(b) * b_y1 * (1.0 / tan(b)) * ((2.0 - 2.0 * nu) *
    cos(b) - ((sqrt((b_y1 * b_y1 + y2 * y2) + jq_a * jq_a) * cos(b) + y3) + 2.0 *
              a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + kq_a * kq_a) - b_y1 * sin(b))
                    + (y3 + 2.0 * a) * cos(b))) - ((sqrt((b_y1 * b_y1 + y2 * y2)
    + lq_a * lq_a) * cos(b) + y3) + 2.0 * a) * (1.0 / tan(b)) / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + mq_a * mq_a, 1.5) * ((2.0 - 2.0 * nu) * cos(b) -
    ((sqrt((b_y1 * b_y1 + y2 * y2) + nq_a * nq_a) * cos(b) + y3) + 2.0 * a) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + oq_a * oq_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b))) * b_y1) + ((sqrt((b_y1 * b_y1 + y2 * y2) + pq_a * pq_a) * cos
    (b) + y3) + 2.0 * a) * (1.0 / tan(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + qq_a *
    qq_a) * (-1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + rq_a * rq_a) * cos(b) * b_y1 /
             ((sqrt((b_y1 * b_y1 + y2 * y2) + sq_a * sq_a) - b_y1 * sin(b)) +
              (y3 + 2.0 * a) * cos(b)) + ((sqrt((b_y1 * b_y1 + y2 * y2) + tq_a *
    tq_a) * cos(b) + y3) + 2.0 * a) / (uq_a * uq_a) * (1.0 / sqrt((b_y1 * b_y1 +
    y2 * y2) + vq_a * vq_a) * b_y1 - sin(b)))) - a / powd_snf((b_y1 * b_y1 +
    y2 * y2) + wq_a * wq_a, 1.5) * ((sin(b) - (y3 + 2.0 * a) * (b_y1 * cos(b) +
    (y3 + 2.0 * a) * sin(b)) / ((b_y1 * b_y1 + y2 * y2) + xq_a * xq_a)) - (b_y1 *
    cos(b) + (y3 + 2.0 * a) * sin(b)) * ((sqrt((b_y1 * b_y1 + y2 * y2) + yq_a *
    yq_a) * cos(b) + y3) + 2.0 * a) / sqrt((b_y1 * b_y1 + y2 * y2) + ar_a * ar_a)
    / ((sqrt((b_y1 * b_y1 + y2 * y2) + br_a * br_a) - b_y1 * sin(b)) + (y3 + 2.0
    * a) * cos(b))) * b_y1) + a / sqrt((b_y1 * b_y1 + y2 * y2) + cr_a * cr_a) *
             (((((-(y3 + 2.0 * a) * cos(b) / ((b_y1 * b_y1 + y2 * y2) + dr_a *
    dr_a) + 2.0 * (y3 + 2.0 * a) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) /
                  (er_a * er_a) * b_y1) - cos(b) * ((sqrt((b_y1 * b_y1 + y2 * y2)
    + fr_a * fr_a) * cos(b) + y3) + 2.0 * a) / sqrt((b_y1 * b_y1 + y2 * y2) +
    gr_a * gr_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + hr_a * hr_a) - b_y1 * sin(b))
                    + (y3 + 2.0 * a) * cos(b))) - (b_y1 * cos(b) + (y3 + 2.0 * a)
    * sin(b)) / ((b_y1 * b_y1 + y2 * y2) + ir_a * ir_a) * cos(b) * b_y1 / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + jr_a * jr_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b))) + (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * ((sqrt((b_y1 * b_y1 +
    y2 * y2) + kr_a * kr_a) * cos(b) + y3) + 2.0 * a) / powd_snf((b_y1 * b_y1
    + y2 * y2) + lr_a * lr_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + mr_a *
    mr_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * b_y1) + (b_y1 * cos(b) +
    (y3 + 2.0 * a) * sin(b)) * ((sqrt((b_y1 * b_y1 + y2 * y2) + nr_a * nr_a) *
    cos(b) + y3) + 2.0 * a) / sqrt((b_y1 * b_y1 + y2 * y2) + or_a * or_a) /
              (pr_a * pr_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + qr_a * qr_a)
    * b_y1 - sin(b))))) / M_PI / (1.0 - nu))) + 0.5 * B3 * (0.125 *
    ((2.0 - 2.0 * nu) * (((-y2 / (rr_a * rr_a) * cos(b) / (1.0 + y2 * y2 / (sr_a
    * sr_a)) + (y2 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) / (b_y1 *
    (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b)) * b_y1 - y2 * sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) * sin(b) / (tr_a * tr_a) * (2.0 * b_y1 * cos(b) -
    y3 * sin(b))) / (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * (k_x *
    k_x) / (ur_a * ur_a))) + y2 / (vr_a * vr_a) * cos(b) / (1.0 + y2 * y2 /
    (wr_a * wr_a))) - (y2 / sqrt((b_y1 * b_y1 + y2 * y2) + xr_a * xr_a) * sin(b)
                       / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 *
    y2 * cos(b)) * b_y1 - y2 * sqrt((b_y1 * b_y1 + y2 * y2) + yr_a * yr_a) * sin
                       (b) / (as_a * as_a) * (2.0 * b_y1 * cos(b) + (y3 + 2.0 *
    a) * sin(b))) / (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2) + bs_a * bs_a) *
                     (l_x * l_x) / (cs_a * cs_a))) + y2 * sin(b) * (((((1.0 /
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b) * b_y1 / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) - (sqrt((b_y1 * b_y1 +
    y2 * y2) + y3 * y3) * cos(b) - y3) / powd_snf((b_y1 * b_y1 + y2 * y2) +
    y3 * y3, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) -
                     y3 * cos(b)) * b_y1) - (sqrt((b_y1 * b_y1 + y2 * y2) + y3 *
    y3) * cos(b) - y3) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (ds_a * ds_a)
    * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * b_y1 - sin(b))) - 1.0 /
    ((b_y1 * b_y1 + y2 * y2) + es_a * es_a) * cos(b) * b_y1 / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + fs_a * fs_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)))
    + ((sqrt((b_y1 * b_y1 + y2 * y2) + gs_a * gs_a) * cos(b) + y3) + 2.0 * a) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + hs_a * hs_a, 1.5) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + is_a * is_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    b_y1) + ((sqrt((b_y1 * b_y1 + y2 * y2) + js_a * js_a) * cos(b) + y3) + 2.0 *
             a) / sqrt((b_y1 * b_y1 + y2 * y2) + ks_a * ks_a) / (ls_a * ls_a) *
    (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + ms_a * ms_a) * b_y1 - sin(b)))) /
    M_PI / (1.0 - nu) + 0.25 * ((((((2.0 - 2.0 * nu) * ((y2 /
    (b_y1 * b_y1) / (1.0 + y2 * y2 / (b_y1 * b_y1)) - y2 / (ns_a * ns_a) * cos(b)
    / (1.0 + y2 * y2 / (os_a * os_a))) + (y2 / sqrt((b_y1 * b_y1 + y2 * y2) +
    ps_a * ps_a) * sin(b) / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) +
    y2 * y2 * cos(b)) * b_y1 - y2 * sqrt((b_y1 * b_y1 + y2 * y2) + qs_a * qs_a) *
    sin(b) / (rs_a * rs_a) * (2.0 * b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b))) /
    (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2) + ss_a * ss_a) * (m_x * m_x) /
     (ts_a * ts_a))) - (2.0 - 2.0 * nu) * y2 * sin(b) / (us_a * us_a) * (cos(b)
    + a / sqrt((b_y1 * b_y1 + y2 * y2) + vs_a * vs_a)) * (1.0 / sqrt((b_y1 *
    b_y1 + y2 * y2) + ws_a * ws_a) * b_y1 - sin(b))) - (2.0 - 2.0 * nu) * y2 *
    sin(b) / ((sqrt((b_y1 * b_y1 + y2 * y2) + xs_a * xs_a) - b_y1 * sin(b)) +
              (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 * y2)
    + ys_a * ys_a, 1.5) * b_y1) - y2 * (y3 + a) * sin(b) / powd_snf((b_y1 *
    b_y1 + y2 * y2) + at_a * at_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + bt_a *
    bt_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((1.0 + ((sqrt((b_y1 *
    b_y1 + y2 * y2) + ct_a * ct_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + dt_a * dt_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + et_a * et_a))) + a * (y3 + 2.0 *
    a) / ((b_y1 * b_y1 + y2 * y2) + ft_a * ft_a)) * b_y1) - y2 * (y3 + a) * sin
    (b) / sqrt((b_y1 * b_y1 + y2 * y2) + gt_a * gt_a) / (ht_a * ht_a) * ((1.0 +
    ((sqrt((b_y1 * b_y1 + y2 * y2) + it_a * it_a) * cos(b) + y3) + 2.0 * a) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + jt_a * jt_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + kt_a * kt_a))) +
    a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) + lt_a * lt_a)) * (1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + mt_a * mt_a) * b_y1 - sin(b))) + y2 * (y3 + a) *
    sin(b) / sqrt((b_y1 * b_y1 + y2 * y2) + nt_a * nt_a) / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + ot_a * ot_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((1.0
    / sqrt((b_y1 * b_y1 + y2 * y2) + pt_a * pt_a) * cos(b) * b_y1 / ((sqrt((b_y1
    * b_y1 + y2 * y2) + qt_a * qt_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))
    * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + rt_a * rt_a)) - ((sqrt((b_y1 *
    b_y1 + y2 * y2) + st_a * st_a) * cos(b) + y3) + 2.0 * a) / (tt_a * tt_a) *
    (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + ut_a * ut_a)) * (1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + vt_a * vt_a) * b_y1 - sin(b))) - ((sqrt((b_y1 *
    b_y1 + y2 * y2) + wt_a * wt_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + xt_a * xt_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    a / powd_snf((b_y1 * b_y1 + y2 * y2) + yt_a * yt_a, 1.5) * b_y1) - 2.0 *
    a * (y3 + 2.0 * a) / (au_a * au_a) * b_y1)) / M_PI / (1.0 - nu));
  b_a = y3 + 2.0 * a;
  c_a = y3 + 2.0 * a;
  d_a = y3 + 2.0 * a;
  e_a = y3 + 2.0 * a;
  f_a = sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3;
  g_a = y3 + 2.0 * a;
  h_a = y3 + 2.0 * a;
  i_a = y3 + 2.0 * a;
  j_a = y3 + 2.0 * a;
  j_a = (sqrt((b_y1 * b_y1 + y2 * y2) + j_a * j_a) + y3) + 2.0 * a;
  k_a = y3 + 2.0 * a;
  l_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  m_a = y3 + 2.0 * a;
  n_a = y3 + 2.0 * a;
  o_a = y3 + 2.0 * a;
  p_a = y3 + 2.0 * a;
  p_a = (sqrt((b_y1 * b_y1 + y2 * y2) + p_a * p_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b);
  q_a = y3 + 2.0 * a;
  x = 1.0 / tan(b);
  r_a = y3 + 2.0 * a;
  s_a = y3 + 2.0 * a;
  b_x = 1.0 / tan(b);
  t_a = y3 + 2.0 * a;
  u_a = y3 + 2.0 * a;
  v_a = y3 + 2.0 * a;
  v_a = (sqrt((b_y1 * b_y1 + y2 * y2) + v_a * v_a) + y3) + 2.0 * a;
  w_a = y3 + 2.0 * a;
  x_a = y3 + 2.0 * a;
  y_a = y3 + 2.0 * a;
  ab_a = y3 + 2.0 * a;
  bb_a = y3 + 2.0 * a;
  cb_a = y3 + 2.0 * a;
  db_a = y3 + 2.0 * a;
  db_a = (sqrt((b_y1 * b_y1 + y2 * y2) + db_a * db_a) + y3) + 2.0 * a;
  eb_a = y3 + 2.0 * a;
  fb_a = y3 + 2.0 * a;
  gb_a = y3 + 2.0 * a;
  hb_a = y3 + 2.0 * a;
  ib_a = y3 + 2.0 * a;
  jb_a = y3 + 2.0 * a;
  kb_a = y3 + 2.0 * a;
  kb_a = (sqrt((b_y1 * b_y1 + y2 * y2) + kb_a * kb_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  lb_a = y3 + 2.0 * a;
  mb_a = y3 + 2.0 * a;
  nb_a = y3 + 2.0 * a;
  ob_a = y3 + 2.0 * a;
  pb_a = y3 + 2.0 * a;
  qb_a = y3 + 2.0 * a;
  rb_a = y3 + 2.0 * a;
  sb_a = y3 + 2.0 * a;
  tb_a = y3 + 2.0 * a;
  ub_a = y3 + 2.0 * a;
  vb_a = y3 + 2.0 * a;
  wb_a = y3 + 2.0 * a;
  xb_a = y3 + 2.0 * a;
  xb_a = (sqrt((b_y1 * b_y1 + y2 * y2) + xb_a * xb_a) + y3) + 2.0 * a;
  yb_a = y3 + 2.0 * a;
  ac_a = y3 + 2.0 * a;
  bc_a = y3 + 2.0 * a;
  cc_a = y3 + 2.0 * a;
  dc_a = y3 + 2.0 * a;
  ec_a = y3 + 2.0 * a;
  fc_a = y3 + 2.0 * a;
  gc_a = y3 + 2.0 * a;
  hc_a = y3 + 2.0 * a;
  ic_a = y3 + 2.0 * a;
  jc_a = y3 + 2.0 * a;
  kc_a = y3 + 2.0 * a;
  lc_a = y3 + 2.0 * a;
  lc_a = (sqrt((b_y1 * b_y1 + y2 * y2) + lc_a * lc_a) + y3) + 2.0 * a;
  mc_a = y3 + 2.0 * a;
  nc_a = y3 + 2.0 * a;
  oc_a = y3 + 2.0 * a;
  oc_a = (b_y1 * b_y1 + y2 * y2) + oc_a * oc_a;
  pc_a = y3 + 2.0 * a;
  qc_a = y3 + 2.0 * a;
  rc_a = y3 + 2.0 * a;
  c_x = cos(b);
  sc_a = y3 + 2.0 * a;
  tc_a = y3 + 2.0 * a;
  uc_a = y3 + 2.0 * a;
  vc_a = y3 + 2.0 * a;
  d_x = cos(b);
  wc_a = y3 + 2.0 * a;
  xc_a = y3 + 2.0 * a;
  yc_a = y3 + 2.0 * a;
  yc_a = (sqrt((b_y1 * b_y1 + y2 * y2) + yc_a * yc_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  e_x = cos(b);
  ad_a = y3 + 2.0 * a;
  bd_a = y3 + 2.0 * a;
  cd_a = y3 + 2.0 * a;
  dd_a = y3 + 2.0 * a;
  f_x = cos(b);
  ed_a = y3 + 2.0 * a;
  fd_a = y3 + 2.0 * a;
  gd_a = y3 + 2.0 * a;
  hd_a = y3 + 2.0 * a;
  id_a = y3 + 2.0 * a;
  jd_a = y3 + 2.0 * a;
  kd_a = y3 + 2.0 * a;
  ld_a = y3 + 2.0 * a;
  md_a = y3 + 2.0 * a;
  nd_a = y3 + 2.0 * a;
  od_a = y3 + 2.0 * a;
  g_x = cos(b);
  pd_a = y3 + 2.0 * a;
  qd_a = y3 + 2.0 * a;
  rd_a = y3 + 2.0 * a;
  sd_a = y3 + 2.0 * a;
  sd_a = (sqrt((b_y1 * b_y1 + y2 * y2) + sd_a * sd_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  h_x = cos(b);
  td_a = y3 + 2.0 * a;
  ud_a = y3 + 2.0 * a;
  vd_a = y3 + 2.0 * a;
  wd_a = y3 + 2.0 * a;
  xd_a = y3 + 2.0 * a;
  yd_a = y3 + 2.0 * a;
  ae_a = y3 + 2.0 * a;
  be_a = y3 + 2.0 * a;
  ce_a = y3 + 2.0 * a;
  de_a = y3 + 2.0 * a;
  ee_a = y3 + 2.0 * a;
  fe_a = b_y1 * cos(b) - y3 * sin(b);
  ge_a = b_y1 * cos(b) - y3 * sin(b);
  i_x = sin(b);
  he_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  j_x = sin(b);
  ie_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  je_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  ke_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  le_a = y3 + 2.0 * a;
  me_a = y3 + 2.0 * a;
  k_x = sin(b);
  ne_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  oe_a = y3 + 2.0 * a;
  l_x = sin(b);
  pe_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  qe_a = sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - y3;
  re_a = y3 + 2.0 * a;
  se_a = y3 + 2.0 * a;
  te_a = y3 + 2.0 * a;
  ue_a = y3 + 2.0 * a;
  ue_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ue_a * ue_a) + y3) + 2.0 * a;
  ve_a = y3 + 2.0 * a;
  we_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  xe_a = y3 + 2.0 * a;
  ye_a = y3 + 2.0 * a;
  af_a = y3 + 2.0 * a;
  bf_a = y3 + 2.0 * a;
  cf_a = y3 + 2.0 * a;
  df_a = y3 + 2.0 * a;
  df_a = (sqrt((b_y1 * b_y1 + y2 * y2) + df_a * df_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  ef_a = y3 + 2.0 * a;
  ff_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  gf_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  hf_a = y3 + 2.0 * a;
  if_a = y3 + 2.0 * a;
  m_x = sin(b);
  jf_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  kf_a = y3 + 2.0 * a;
  n_x = sin(b);
  lf_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  o_x = 1.0 / tan(b);
  mf_a = y3 + 2.0 * a;
  mf_a = (sqrt((b_y1 * b_y1 + y2 * y2) + mf_a * mf_a) + y3) + 2.0 * a;
  nf_a = y3 + 2.0 * a;
  of_a = y3 + 2.0 * a;
  pf_a = y3 + 2.0 * a;
  qf_a = y3 + 2.0 * a;
  rf_a = y3 + 2.0 * a;
  sf_a = y3 + 2.0 * a;
  tf_a = y3 + 2.0 * a;
  tf_a = (sqrt((b_y1 * b_y1 + y2 * y2) + tf_a * tf_a) + y3) + 2.0 * a;
  uf_a = y3 + 2.0 * a;
  vf_a = y3 + 2.0 * a;
  wf_a = y3 + 2.0 * a;
  xf_a = y3 + 2.0 * a;
  yf_a = y3 + 2.0 * a;
  yf_a = (sqrt((b_y1 * b_y1 + y2 * y2) + yf_a * yf_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  ag_a = y3 + 2.0 * a;
  bg_a = y3 + 2.0 * a;
  cg_a = y3 + 2.0 * a;
  dg_a = y3 + 2.0 * a;
  eg_a = y3 + 2.0 * a;
  fg_a = y3 + 2.0 * a;
  gg_a = y3 + 2.0 * a;
  hg_a = y3 + 2.0 * a;
  ig_a = y3 + 2.0 * a;
  jg_a = y3 + 2.0 * a;
  kg_a = y3 + 2.0 * a;
  lg_a = y3 + 2.0 * a;
  mg_a = y3 + 2.0 * a;
  ng_a = y3 + 2.0 * a;
  og_a = y3 + 2.0 * a;
  pg_a = y3 + 2.0 * a;
  qg_a = y3 + 2.0 * a;
  rg_a = y3 + 2.0 * a;
  sg_a = y3 + 2.0 * a;
  tg_a = y3 + 2.0 * a;
  tg_a = (sqrt((b_y1 * b_y1 + y2 * y2) + tg_a * tg_a) + y3) + 2.0 * a;
  ug_a = y3 + 2.0 * a;
  vg_a = y3 + 2.0 * a;
  wg_a = y3 + 2.0 * a;
  xg_a = y3 + 2.0 * a;
  yg_a = y3 + 2.0 * a;
  ah_a = y3 + 2.0 * a;
  bh_a = y3 + 2.0 * a;
  ch_a = y3 + 2.0 * a;
  ch_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ch_a * ch_a) + y3) + 2.0 * a;
  dh_a = y3 + 2.0 * a;
  eh_a = y3 + 2.0 * a;
  fh_a = y3 + 2.0 * a;
  gh_a = y3 + 2.0 * a;
  hh_a = y3 + 2.0 * a;
  ih_a = y3 + 2.0 * a;
  jh_a = y3 + 2.0 * a;
  jh_a = (sqrt((b_y1 * b_y1 + y2 * y2) + jh_a * jh_a) + y3) + 2.0 * a;
  kh_a = y3 + 2.0 * a;
  lh_a = y3 + 2.0 * a;
  mh_a = y3 + 2.0 * a;
  nh_a = y3 + 2.0 * a;
  oh_a = y3 + 2.0 * a;
  ph_a = y3 + 2.0 * a;
  qh_a = y3 + 2.0 * a;
  rh_a = y3 + 2.0 * a;
  sh_a = y3 + 2.0 * a;
  th_a = y3 + 2.0 * a;
  uh_a = y3 + 2.0 * a;
  vh_a = y3 + 2.0 * a;
  wh_a = y3 + 2.0 * a;
  xh_a = y3 + 2.0 * a;
  yh_a = y3 + 2.0 * a;
  yh_a = (sqrt((b_y1 * b_y1 + y2 * y2) + yh_a * yh_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  ai_a = y3 + 2.0 * a;
  bi_a = y3 + 2.0 * a;
  ci_a = y3 + 2.0 * a;
  di_a = y3 + 2.0 * a;
  ei_a = y3 + 2.0 * a;
  fi_a = y3 + 2.0 * a;
  gi_a = y3 + 2.0 * a;
  hi_a = y3 + 2.0 * a;
  ii_a = y3 + 2.0 * a;
  ji_a = y3 + 2.0 * a;
  ki_a = y3 + 2.0 * a;
  li_a = y3 + 2.0 * a;
  li_a = (sqrt((b_y1 * b_y1 + y2 * y2) + li_a * li_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  mi_a = y3 + 2.0 * a;
  ni_a = y3 + 2.0 * a;
  oi_a = y3 + 2.0 * a;
  pi_a = y3 + 2.0 * a;
  qi_a = y3 + 2.0 * a;
  ri_a = y3 + 2.0 * a;
  si_a = y3 + 2.0 * a;
  si_a = (b_y1 * b_y1 + y2 * y2) + si_a * si_a;
  ti_a = y3 + 2.0 * a;
  ui_a = y3 + 2.0 * a;
  vi_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  wi_a = y3 + 2.0 * a;
  xi_a = y3 + 2.0 * a;
  yi_a = y3 + 2.0 * a;
  aj_a = y3 + 2.0 * a;
  aj_a = (sqrt((b_y1 * b_y1 + y2 * y2) + aj_a * aj_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  bj_a = y3 + 2.0 * a;
  cj_a = y3 + 2.0 * a;
  dj_a = y3 + 2.0 * a;
  ej_a = y3 + 2.0 * a;
  ej_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ej_a * ej_a) + y3) + 2.0 * a;
  fj_a = y3 + 2.0 * a;
  gj_a = y3 + 2.0 * a;
  hj_a = y3 + 2.0 * a;
  ij_a = y3 + 2.0 * a;
  jj_a = y3 + 2.0 * a;
  kj_a = y3 + 2.0 * a;
  lj_a = y3 + 2.0 * a;
  lj_a = (sqrt((b_y1 * b_y1 + y2 * y2) + lj_a * lj_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  mj_a = y3 + 2.0 * a;
  nj_a = y3 + 2.0 * a;
  oj_a = y3 + 2.0 * a;
  pj_a = y3 + 2.0 * a;
  qj_a = y3 + 2.0 * a;
  rj_a = y3 + 2.0 * a;
  sj_a = y3 + 2.0 * a;
  tj_a = y3 + 2.0 * a;
  uj_a = y3 + 2.0 * a;
  vj_a = y3 + 2.0 * a;
  wj_a = y3 + 2.0 * a;
  xj_a = y3 + 2.0 * a;
  xj_a = (b_y1 * b_y1 + y2 * y2) + xj_a * xj_a;
  yj_a = y3 + 2.0 * a;
  yj_a = (sqrt((b_y1 * b_y1 + y2 * y2) + yj_a * yj_a) + y3) + 2.0 * a;
  ak_a = y3 + 2.0 * a;
  bk_a = y3 + 2.0 * a;
  ck_a = y3 + 2.0 * a;
  dk_a = y3 + 2.0 * a;
  ek_a = y3 + 2.0 * a;
  fk_a = y3 + 2.0 * a;
  gk_a = y3 + 2.0 * a;
  hk_a = y3 + 2.0 * a;
  ik_a = y3 + 2.0 * a;
  jk_a = y3 + 2.0 * a;
  jk_a = (sqrt((b_y1 * b_y1 + y2 * y2) + jk_a * jk_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  kk_a = y3 + 2.0 * a;
  lk_a = y3 + 2.0 * a;
  mk_a = y3 + 2.0 * a;
  nk_a = y3 + 2.0 * a;
  ok_a = y3 + 2.0 * a;
  pk_a = y3 + 2.0 * a;
  qk_a = y3 + 2.0 * a;
  rk_a = y3 + 2.0 * a;
  sk_a = y3 + 2.0 * a;
  tk_a = y3 + 2.0 * a;
  uk_a = y3 + 2.0 * a;
  vk_a = y3 + 2.0 * a;
  wk_a = y3 + 2.0 * a;
  xk_a = y3 + 2.0 * a;
  yk_a = y3 + 2.0 * a;
  al_a = y3 + 2.0 * a;
  bl_a = y3 + 2.0 * a;
  bl_a = (b_y1 * b_y1 + y2 * y2) + bl_a * bl_a;
  cl_a = y3 + 2.0 * a;
  dl_a = y3 + 2.0 * a;
  el_a = y3 + 2.0 * a;
  fl_a = y3 + 2.0 * a;
  gl_a = y3 + 2.0 * a;
  hl_a = y3 + 2.0 * a;
  hl_a = (sqrt((b_y1 * b_y1 + y2 * y2) + hl_a * hl_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  il_a = y3 + 2.0 * a;
  jl_a = y3 + 2.0 * a;
  kl_a = y3 + 2.0 * a;
  ll_a = y3 + 2.0 * a;
  ml_a = y3 + 2.0 * a;
  nl_a = y3 + 2.0 * a;
  ol_a = y3 + 2.0 * a;
  pl_a = y3 + 2.0 * a;
  ql_a = y3 + 2.0 * a;
  rl_a = y3 + 2.0 * a;
  sl_a = y3 + 2.0 * a;
  tl_a = y3 + 2.0 * a;
  ul_a = y3 + 2.0 * a;
  vl_a = y3 + 2.0 * a;
  wl_a = y3 + 2.0 * a;
  xl_a = y3 + 2.0 * a;
  yl_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  am_a = y3 + 2.0 * a;
  bm_a = y3 + 2.0 * a;
  cm_a = y3 + 2.0 * a;
  dm_a = y3 + 2.0 * a;
  em_a = y3 + 2.0 * a;
  fm_a = y3 + 2.0 * a;
  gm_a = y3 + 2.0 * a;
  hm_a = y3 + 2.0 * a;
  hm_a = (sqrt((b_y1 * b_y1 + y2 * y2) + hm_a * hm_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  im_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  jm_a = y3 + 2.0 * a;
  km_a = y3 + 2.0 * a;
  lm_a = y3 + 2.0 * a;
  mm_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  nm_a = y3 + 2.0 * a;
  p_x = sin(b);
  om_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  pm_a = y3 + 2.0 * a;
  qm_a = y3 + 2.0 * a;
  rm_a = y3 + 2.0 * a;
  rm_a = (sqrt((b_y1 * b_y1 + y2 * y2) + rm_a * rm_a) + y3) + 2.0 * a;
  sm_a = y3 + 2.0 * a;
  tm_a = y3 + 2.0 * a;
  um_a = y3 + 2.0 * a;
  vm_a = y3 + 2.0 * a;
  wm_a = y3 + 2.0 * a;
  xm_a = y3 + 2.0 * a;
  ym_a = y3 + 2.0 * a;
  ym_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ym_a * ym_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  an_a = y3 + 2.0 * a;
  bn_a = y3 + 2.0 * a;
  cn_a = y3 + 2.0 * a;
  dn_a = y3 + 2.0 * a;
  en_a = y3 + 2.0 * a;
  fn_a = y3 + 2.0 * a;
  gn_a = y3 + 2.0 * a;
  hn_a = y3 + 2.0 * a;
  in_a = y3 + 2.0 * a;
  jn_a = y3 + 2.0 * a;
  kn_a = y3 + 2.0 * a;
  ln_a = y3 + 2.0 * a;
  ln_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ln_a * ln_a) + y3) + 2.0 * a;
  mn_a = y3 + 2.0 * a;
  nn_a = y3 + 2.0 * a;
  nn_a = (b_y1 * b_y1 + y2 * y2) + nn_a * nn_a;
  on_a = y3 + 2.0 * a;
  pn_a = y3 + 2.0 * a;
  qn_a = y3 + 2.0 * a;
  rn_a = y3 + 2.0 * a;
  sn_a = y3 + 2.0 * a;
  tn_a = y3 + 2.0 * a;
  un_a = y3 + 2.0 * a;
  vn_a = y3 + 2.0 * a;
  wn_a = y3 + 2.0 * a;
  xn_a = y3 + 2.0 * a;
  yn_a = y3 + 2.0 * a;
  ao_a = y3 + 2.0 * a;
  bo_a = y3 + 2.0 * a;
  co_a = y3 + 2.0 * a;
  co_a = (sqrt((b_y1 * b_y1 + y2 * y2) + co_a * co_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  do_a = y3 + 2.0 * a;
  eo_a = y3 + 2.0 * a;
  fo_a = y3 + 2.0 * a;
  go_a = y3 + 2.0 * a;
  ho_a = y3 + 2.0 * a;
  io_a = y3 + 2.0 * a;
  jo_a = y3 + 2.0 * a;
  ko_a = y3 + 2.0 * a;
  lo_a = y3 + 2.0 * a;
  mo_a = y3 + 2.0 * a;
  no_a = y3 + 2.0 * a;
  no_a = (sqrt((b_y1 * b_y1 + y2 * y2) + no_a * no_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  oo_a = y3 + 2.0 * a;
  po_a = y3 + 2.0 * a;
  qo_a = y3 + 2.0 * a;
  ro_a = y3 + 2.0 * a;
  so_a = y3 + 2.0 * a;
  to_a = y3 + 2.0 * a;
  to_a = (b_y1 * b_y1 + y2 * y2) + to_a * to_a;
  uo_a = y3 + 2.0 * a;
  vo_a = y3 + 2.0 * a;
  wo_a = y3 + 2.0 * a;
  xo_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  yo_a = y3 + 2.0 * a;
  ap_a = y3 + 2.0 * a;
  bp_a = y3 + 2.0 * a;
  cp_a = y3 + 2.0 * a;
  dp_a = y3 + 2.0 * a;
  ep_a = y3 + 2.0 * a;
  fp_a = y3 + 2.0 * a;
  gp_a = y3 + 2.0 * a;
  gp_a = (sqrt((b_y1 * b_y1 + y2 * y2) + gp_a * gp_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  hp_a = y3 + 2.0 * a;
  ip_a = y3 + 2.0 * a;
  jp_a = y3 + 2.0 * a;
  kp_a = y3 + 2.0 * a;
  lp_a = y3 + 2.0 * a;
  lp_a = (sqrt((b_y1 * b_y1 + y2 * y2) + lp_a * lp_a) + y3) + 2.0 * a;
  mp_a = y3 + 2.0 * a;
  np_a = y3 + 2.0 * a;
  op_a = y3 + 2.0 * a;
  pp_a = y3 + 2.0 * a;
  qp_a = y3 + 2.0 * a;
  qp_a = (sqrt((b_y1 * b_y1 + y2 * y2) + qp_a * qp_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  rp_a = y3 + 2.0 * a;
  sp_a = y3 + 2.0 * a;
  tp_a = y3 + 2.0 * a;
  up_a = y3 + 2.0 * a;
  vp_a = y3 + 2.0 * a;
  wp_a = y3 + 2.0 * a;
  xp_a = y3 + 2.0 * a;
  yp_a = y3 + 2.0 * a;
  aq_a = y3 + 2.0 * a;
  aq_a = (sqrt((b_y1 * b_y1 + y2 * y2) + aq_a * aq_a) + y3) + 2.0 * a;
  bq_a = y3 + 2.0 * a;
  cq_a = y3 + 2.0 * a;
  cq_a = (b_y1 * b_y1 + y2 * y2) + cq_a * cq_a;
  dq_a = y3 + 2.0 * a;
  dq_a = (sqrt((b_y1 * b_y1 + y2 * y2) + dq_a * dq_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  eq_a = y3 + 2.0 * a;
  fq_a = y3 + 2.0 * a;
  gq_a = y3 + 2.0 * a;
  hq_a = y3 + 2.0 * a;
  iq_a = y3 + 2.0 * a;
  jq_a = y3 + 2.0 * a;
  kq_a = y3 + 2.0 * a;
  lq_a = y3 + 2.0 * a;
  mq_a = y3 + 2.0 * a;
  nq_a = y3 + 2.0 * a;
  oq_a = y3 + 2.0 * a;
  pq_a = y3 + 2.0 * a;
  qq_a = y3 + 2.0 * a;
  rq_a = y3 + 2.0 * a;
  sq_a = y3 + 2.0 * a;
  tq_a = y3 + 2.0 * a;
  uq_a = y3 + 2.0 * a;
  vq_a = y3 + 2.0 * a;
  wq_a = y3 + 2.0 * a;
  xq_a = y3 + 2.0 * a;
  yq_a = y3 + 2.0 * a;
  ar_a = y3 + 2.0 * a;
  br_a = y3 + 2.0 * a;
  cr_a = y3 + 2.0 * a;
  cr_a = (sqrt((b_y1 * b_y1 + y2 * y2) + cr_a * cr_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  dr_a = y3 + 2.0 * a;
  er_a = y3 + 2.0 * a;
  fr_a = y3 + 2.0 * a;
  gr_a = y3 + 2.0 * a;
  hr_a = y3 + 2.0 * a;
  ir_a = y3 + 2.0 * a;
  jr_a = y3 + 2.0 * a;
  kr_a = y3 + 2.0 * a;
  kr_a = (b_y1 * b_y1 + y2 * y2) + kr_a * kr_a;
  lr_a = y3 + 2.0 * a;
  mr_a = y3 + 2.0 * a;
  nr_a = y3 + 2.0 * a;
  or_a = y3 + 2.0 * a;
  pr_a = y3 + 2.0 * a;
  qr_a = y3 + 2.0 * a;
  rr_a = y3 + 2.0 * a;
  sr_a = y3 + 2.0 * a;
  sr_a = (sqrt((b_y1 * b_y1 + y2 * y2) + sr_a * sr_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  tr_a = b_y1 * cos(b) - y3 * sin(b);
  ur_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  lu_a = sin(b);
  vr_a = b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b);
  wr_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  xr_a = y3 + 2.0 * a;
  yr_a = y3 + 2.0 * a;
  as_a = y3 + 2.0 * a;
  bs_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  cs_a = y3 + 2.0 * a;
  mu_a = sin(b);
  ds_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  es_a = y3 + 2.0 * a;
  fs_a = y3 + 2.0 * a;
  gs_a = y3 + 2.0 * a;
  hs_a = (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b);
  is_a = y3 + 2.0 * a;
  js_a = y3 + 2.0 * a;
  ks_a = y3 + 2.0 * a;
  ls_a = y3 + 2.0 * a;
  ms_a = y3 + 2.0 * a;
  ns_a = y3 + 2.0 * a;
  os_a = y3 + 2.0 * a;
  ps_a = y3 + 2.0 * a;
  ps_a = (sqrt((b_y1 * b_y1 + y2 * y2) + ps_a * ps_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  qs_a = b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b);
  rs_a = y3 + 2.0 * a;
  ss_a = y3 + 2.0 * a;
  ts_a = y3 + 2.0 * a;
  us_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  vs_a = y3 + 2.0 * a;
  nu_a = sin(b);
  ws_a = b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b);
  xs_a = y3 + 2.0 * a;
  ys_a = y3 + 2.0 * a;
  at_a = y3 + 2.0 * a;
  at_a = (sqrt((b_y1 * b_y1 + y2 * y2) + at_a * at_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  bt_a = y3 + 2.0 * a;
  ct_a = y3 + 2.0 * a;
  dt_a = y3 + 2.0 * a;
  et_a = y3 + 2.0 * a;
  ft_a = y3 + 2.0 * a;
  gt_a = y3 + 2.0 * a;
  ht_a = y3 + 2.0 * a;
  it_a = y3 + 2.0 * a;
  jt_a = y3 + 2.0 * a;
  kt_a = y3 + 2.0 * a;
  lt_a = y3 + 2.0 * a;
  mt_a = y3 + 2.0 * a;
  nt_a = y3 + 2.0 * a;
  ot_a = y3 + 2.0 * a;
  pt_a = y3 + 2.0 * a;
  qt_a = y3 + 2.0 * a;
  rt_a = y3 + 2.0 * a;
  st_a = y3 + 2.0 * a;
  st_a = (sqrt((b_y1 * b_y1 + y2 * y2) + st_a * st_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  tt_a = y3 + 2.0 * a;
  ut_a = y3 + 2.0 * a;
  vt_a = y3 + 2.0 * a;
  wt_a = y3 + 2.0 * a;
  xt_a = y3 + 2.0 * a;
  yt_a = y3 + 2.0 * a;
  au_a = y3 + 2.0 * a;
  bu_a = y3 + 2.0 * a;
  cu_a = y3 + 2.0 * a;
  du_a = y3 + 2.0 * a;
  eu_a = y3 + 2.0 * a;
  eu_a = (sqrt((b_y1 * b_y1 + y2 * y2) + eu_a * eu_a) - b_y1 * sin(b)) + (y3 +
    2.0 * a) * cos(b);
  fu_a = y3 + 2.0 * a;
  gu_a = y3 + 2.0 * a;
  hu_a = y3 + 2.0 * a;
  iu_a = y3 + 2.0 * a;
  ju_a = y3 + 2.0 * a;
  ku_a = y3 + 2.0 * a;
  ku_a = (b_y1 * b_y1 + y2 * y2) + ku_a * ku_a;
  *e23 = ((((0.5 * B1 * (0.125 * ((1.0 - 2.0 * nu) * (((1.0 / sqrt((b_y1 * b_y1
    + y2 * y2) + y3 * y3) * y3 - 1.0) / (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3)
    - y3) + (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + b_a * b_a) * (2.0 * y3 + 4.0 *
    a) + 1.0) / ((sqrt((b_y1 * b_y1 + y2 * y2) + c_a * c_a) + y3) + 2.0 * a)) -
    cos(b) * ((1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * y3 - cos(b)) /
              ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 *
               cos(b)) + (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + d_a * d_a) * (2.0
    * y3 + 4.0 * a) + cos(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + e_a * e_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)))) - y2 * y2 * ((((-1.0 /
    powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / (sqrt((b_y1 * b_y1 +
    y2 * y2) + y3 * y3) - y3) * y3 - 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 *
    y3) / (f_a * f_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * y3 -
    1.0)) - 0.5 / powd_snf((b_y1 * b_y1 + y2 * y2) + g_a * g_a, 1.5) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + h_a * h_a) + y3) + 2.0 * a) * (2.0 * y3 + 4.0 * a))
    - 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + i_a * i_a) / (j_a * j_a) * (0.5 /
    sqrt((b_y1 * b_y1 + y2 * y2) + k_a * k_a) * (2.0 * y3 + 4.0 * a) + 1.0)) -
    cos(b) * (((-1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) /
                ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 *
                 cos(b)) * y3 - 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) /
                (l_a * l_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) *
    y3 - cos(b))) - 0.5 / powd_snf((b_y1 * b_y1 + y2 * y2) + m_a * m_a, 1.5) /
               ((sqrt((b_y1 * b_y1 + y2 * y2) + n_a * n_a) - b_y1 * sin(b)) +
                (y3 + 2.0 * a) * cos(b)) * (2.0 * y3 + 4.0 * a)) - 1.0 / sqrt
              ((b_y1 * b_y1 + y2 * y2) + o_a * o_a) / (p_a * p_a) * (0.5 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + q_a * q_a) * (2.0 * y3 + 4.0 * a) + cos(b))))) /
              M_PI / (1.0 - nu) + 0.25 * ((((((((((((((1.0 - 2.0 *
    nu) * (((2.0 - 2.0 * nu) * (x * x) - nu) * (0.5 / sqrt((b_y1 * b_y1 + y2 *
    y2) + r_a * r_a) * (2.0 * y3 + 4.0 * a) + 1.0) / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + s_a * s_a) + y3) + 2.0 * a) - (((2.0 - 2.0 * nu) * (b_x * b_x) + 1.0)
    - 2.0 * nu) * cos(b) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + t_a * t_a) *
    (2.0 * y3 + 4.0 * a) + cos(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + u_a * u_a)
    - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) + (1.0 - 2.0 * nu) / (v_a * v_a)
    * (((b_y1 * (1.0 / tan(b)) * ((1.0 - 2.0 * nu) - a / sqrt((b_y1 * b_y1 + y2 *
    y2) + w_a * w_a)) + nu * (y3 + 2.0 * a)) - a) + y2 * y2 / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + x_a * x_a) + y3) + 2.0 * a) * (nu + a / sqrt((b_y1 * b_y1
    + y2 * y2) + y_a * y_a))) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + ab_a *
    ab_a) * (2.0 * y3 + 4.0 * a) + 1.0)) - (1.0 - 2.0 * nu) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + bb_a * bb_a) + y3) + 2.0 * a) * (((0.5 * a * b_y1 * (1.0 /
    tan(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + cb_a * cb_a, 1.5) * (2.0 *
    y3 + 4.0 * a) + nu) - y2 * y2 / (db_a * db_a) * (nu + a / sqrt((b_y1 * b_y1
    + y2 * y2) + eb_a * eb_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + fb_a *
    fb_a) * (2.0 * y3 + 4.0 * a) + 1.0)) - 0.5 * (y2 * y2) / ((sqrt((b_y1 * b_y1
    + y2 * y2) + gb_a * gb_a) + y3) + 2.0 * a) * a / powd_snf((b_y1 * b_y1 +
    y2 * y2) + hb_a * hb_a, 1.5) * (2.0 * y3 + 4.0 * a))) - (1.0 - 2.0 * nu) *
    sin(b) * (1.0 / tan(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ib_a * ib_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1 * b_y1
    + y2 * y2) + jb_a * jb_a))) + (1.0 - 2.0 * nu) * (b_y1 * cos(b) + (y3 + 2.0 *
    a) * sin(b)) * (1.0 / tan(b)) / (kb_a * kb_a) * (cos(b) + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + lb_a * lb_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + mb_a
    * mb_a) * (2.0 * y3 + 4.0 * a) + cos(b))) + 0.5 * (1.0 - 2.0 * nu) * (b_y1 *
    cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + nb_a * nb_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * a /
    powd_snf((b_y1 * b_y1 + y2 * y2) + ob_a * ob_a, 1.5) * (2.0 * y3 + 4.0 *
    a)) - a / powd_snf((b_y1 * b_y1 + y2 * y2) + pb_a * pb_a, 1.5) * b_y1 *
    (1.0 / tan(b))) + 1.5 * a * b_y1 * (y3 + a) * (1.0 / tan(b)) / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + qb_a * qb_a, 2.5) * (2.0 * y3 + 4.0 * a)) + 1.0 /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + rb_a * rb_a) + y3) + 2.0 * a) * (((-2.0 *
    nu + 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + sb_a * sb_a) * ((1.0 - 2.0 * nu) *
    b_y1 * (1.0 / tan(b)) - a)) + y2 * y2 / sqrt((b_y1 * b_y1 + y2 * y2) + tb_a *
    tb_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ub_a * ub_a) + y3) + 2.0 * a) *
    (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + vb_a * vb_a))) + a * (y2 * y2)
    / powd_snf((b_y1 * b_y1 + y2 * y2) + wb_a * wb_a, 1.5))) - (y3 + a) /
    (xb_a * xb_a) * (((-2.0 * nu + 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + yb_a *
    yb_a) * ((1.0 - 2.0 * nu) * b_y1 * (1.0 / tan(b)) - a)) + y2 * y2 / sqrt
                      ((b_y1 * b_y1 + y2 * y2) + ac_a * ac_a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + bc_a * bc_a) + y3) + 2.0 * a) * (2.0 * nu + a / sqrt((b_y1
    * b_y1 + y2 * y2) + cc_a * cc_a))) + a * (y2 * y2) / powd_snf((b_y1 *
    b_y1 + y2 * y2) + dc_a * dc_a, 1.5)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) +
    ec_a * ec_a) * (2.0 * y3 + 4.0 * a) + 1.0)) + (y3 + a) / ((sqrt((b_y1 * b_y1
    + y2 * y2) + fc_a * fc_a) + y3) + 2.0 * a) * ((((-0.5 / powd_snf((b_y1 *
    b_y1 + y2 * y2) + gc_a * gc_a, 1.5) * ((1.0 - 2.0 * nu) * b_y1 * (1.0 / tan
    (b)) - a) * (2.0 * y3 + 4.0 * a) - 0.5 * (y2 * y2) / powd_snf((b_y1 *
    b_y1 + y2 * y2) + hc_a * hc_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ic_a *
    ic_a) + y3) + 2.0 * a) * (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + jc_a
    * jc_a)) * (2.0 * y3 + 4.0 * a)) - y2 * y2 / sqrt((b_y1 * b_y1 + y2 * y2) +
    kc_a * kc_a) / (lc_a * lc_a) * (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 * y2)
    + mc_a * mc_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + nc_a * nc_a) * (2.0 *
    y3 + 4.0 * a) + 1.0)) - 0.5 * (y2 * y2) / (oc_a * oc_a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + pc_a * pc_a) + y3) + 2.0 * a) * a * (2.0 * y3 + 4.0 * a))
    - 1.5 * a * (y2 * y2) / powd_snf((b_y1 * b_y1 + y2 * y2) + qc_a * qc_a,
    2.5) * (2.0 * y3 + 4.0 * a))) + 1.0 / ((sqrt((b_y1 * b_y1 + y2 * y2) + rc_a *
    rc_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((c_x * c_x - 1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + sc_a * sc_a) * ((1.0 - 2.0 * nu) * (b_y1 *
    cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) + a * cos(b))) + a * (y3
    + 2.0 * a) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + tc_a * tc_a, 1.5)) - 1.0 / sqrt((b_y1 *
    b_y1 + y2 * y2) + uc_a * uc_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + vc_a *
    vc_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (y2 * y2 * (d_x * d_x) -
    a * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + wc_a * wc_a) * ((sqrt((b_y1 * b_y1 + y2 * y2) + xc_a *
    xc_a) * cos(b) + y3) + 2.0 * a)))) - (y3 + a) / (yc_a * yc_a) * (((e_x * e_x
    - 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + ad_a * ad_a) * ((1.0 - 2.0 * nu) *
    (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) + a * cos(b))) +
    a * (y3 + 2.0 * a) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan
    (b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + bd_a * bd_a, 1.5)) - 1.0 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + cd_a * cd_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    dd_a * dd_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (y2 * y2 * (f_x *
    f_x) - a * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) / sqrt
    ((b_y1 * b_y1 + y2 * y2) + ed_a * ed_a) * ((sqrt((b_y1 * b_y1 + y2 * y2) +
    fd_a * fd_a) * cos(b) + y3) + 2.0 * a))) * (0.5 / sqrt((b_y1 * b_y1 + y2 *
    y2) + gd_a * gd_a) * (2.0 * y3 + 4.0 * a) + cos(b))) + (y3 + a) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + hd_a * hd_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b)) * (((((((0.5 / powd_snf((b_y1 * b_y1 + y2 * y2) + id_a * id_a,
    1.5) * ((1.0 - 2.0 * nu) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 /
    tan(b)) + a * cos(b)) * (2.0 * y3 + 4.0 * a) - 1.0 / sqrt((b_y1 * b_y1 + y2 *
    y2) + jd_a * jd_a) * (1.0 - 2.0 * nu) * sin(b) * (1.0 / tan(b))) + a * (b_y1
    * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) / powd_snf((b_y1 *
    b_y1 + y2 * y2) + kd_a * kd_a, 1.5)) + a * (y3 + 2.0 * a) * sin(b) * (1.0 /
    tan(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + ld_a * ld_a, 1.5)) - 1.5 * a
                  * (y3 + 2.0 * a) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) *
                  (1.0 / tan(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + md_a *
    md_a, 2.5) * (2.0 * y3 + 4.0 * a)) + 0.5 / powd_snf((b_y1 * b_y1 + y2 *
    y2) + nd_a * nd_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + od_a * od_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (y2 * y2 * (g_x * g_x) - a *
    (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + pd_a * pd_a) * ((sqrt((b_y1 * b_y1 + y2 * y2) + qd_a *
    qd_a) * cos(b) + y3) + 2.0 * a)) * (2.0 * y3 + 4.0 * a)) + 1.0 / sqrt((b_y1 *
    b_y1 + y2 * y2) + rd_a * rd_a) / (sd_a * sd_a) * (y2 * y2 * (h_x * h_x) - a *
    (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * (1.0 / tan(b)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + td_a * td_a) * ((sqrt((b_y1 * b_y1 + y2 * y2) + ud_a *
    ud_a) * cos(b) + y3) + 2.0 * a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) +
    vd_a * vd_a) * (2.0 * y3 + 4.0 * a) + cos(b))) - 1.0 / sqrt((b_y1 * b_y1 +
    y2 * y2) + wd_a * wd_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + xd_a * xd_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((-a * sin(b) * (1.0 / tan(b)) /
    sqrt((b_y1 * b_y1 + y2 * y2) + yd_a * yd_a) * ((sqrt((b_y1 * b_y1 + y2 * y2)
    + ae_a * ae_a) * cos(b) + y3) + 2.0 * a) + 0.5 * a * (b_y1 * cos(b) + (y3 +
    2.0 * a) * sin(b)) * (1.0 / tan(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) +
    be_a * be_a, 1.5) * ((sqrt((b_y1 * b_y1 + y2 * y2) + ce_a * ce_a) * cos(b) +
    y3) + 2.0 * a) * (2.0 * y3 + 4.0 * a)) - a * (b_y1 * cos(b) + (y3 + 2.0 * a)
    * sin(b)) * (1.0 / tan(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + de_a * de_a) *
    (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + ee_a * ee_a) * cos(b) * (2.0 * y3 +
    4.0 * a) + 1.0)))) / M_PI / (1.0 - nu)) + 0.5 * B2 * (0.125 *
              (((2.0 - 2.0 * nu) * (((y2 / (fe_a * fe_a) * sin(b) / (1.0 + y2 *
    y2 / (ge_a * ge_a)) + (y2 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b)
    / (b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b)) * y3 + y2 * sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * (i_x * i_x) / (he_a * he_a) * b_y1) /
    (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * (j_x * j_x) / (ie_a *
    ie_a))) - y2 / (je_a * je_a) * sin(b) / (1.0 + y2 * y2 / (ke_a * ke_a))) +
    (0.5 * y2 / sqrt((b_y1 * b_y1 + y2 * y2) + le_a * le_a) * sin(b) / (b_y1 *
    (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b)) * (2.0 * y3 +
    4.0 * a) - y2 * sqrt((b_y1 * b_y1 + y2 * y2) + me_a * me_a) * (k_x * k_x) /
     (ne_a * ne_a) * b_y1) / (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2) + oe_a *
    oe_a) * (l_x * l_x) / (pe_a * pe_a))) + b_y1 * y2 * (((-1.0 / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / (sqrt((b_y1 * b_y1 + y2 * y2) +
    y3 * y3) - y3) * y3 - 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (qe_a *
    qe_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * y3 - 1.0)) - 0.5 /
    powd_snf((b_y1 * b_y1 + y2 * y2) + re_a * re_a, 1.5) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + se_a * se_a) + y3) + 2.0 * a) * (2.0 * y3 + 4.0 * a)) -
    1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + te_a * te_a) / (ue_a * ue_a) * (0.5 /
    sqrt((b_y1 * b_y1 + y2 * y2) + ve_a * ve_a) * (2.0 * y3 + 4.0 * a) + 1.0)))
               - y2 * (((((-sin(b) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) -
    (b_y1 * cos(b) - y3 * sin(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 *
    y3, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 *
                cos(b)) * y3) - (b_y1 * cos(b) - y3 * sin(b)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) / (we_a * we_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 *
    y2) + y3 * y3) * y3 - cos(b))) + sin(b) / sqrt((b_y1 * b_y1 + y2 * y2) +
    xe_a * xe_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ye_a * ye_a) - b_y1 * sin(b))
                    + (y3 + 2.0 * a) * cos(b))) - 0.5 * (b_y1 * cos(b) + (y3 +
    2.0 * a) * sin(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + af_a * af_a, 1.5)
                        / ((sqrt((b_y1 * b_y1 + y2 * y2) + bf_a * bf_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * (2.0 * y3 + 4.0 * a)) - (b_y1 * cos(b)
    + (y3 + 2.0 * a) * sin(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + cf_a * cf_a) /
                       (df_a * df_a) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) +
    ef_a * ef_a) * (2.0 * y3 + 4.0 * a) + cos(b)))) / M_PI / (1.0
    - nu) + 0.25 * (((((((((((((((2.0 - 2.0 * nu) * (1.0 - 2.0 * nu) * (-y2 /
    (ff_a * ff_a) * sin(b) / (1.0 + y2 * y2 / (gf_a * gf_a)) + (0.5 * y2 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + hf_a * hf_a) * sin(b) / (b_y1 * (b_y1 * cos(b) +
    (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b)) * (2.0 * y3 + 4.0 * a) - y2 *
    sqrt((b_y1 * b_y1 + y2 * y2) + if_a * if_a) * (m_x * m_x) / (jf_a * jf_a) *
    b_y1) / (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2) + kf_a * kf_a) * (n_x *
    n_x) / (lf_a * lf_a))) * (o_x * o_x) - (1.0 - 2.0 * nu) * y2 / (mf_a * mf_a)
    * (((-1.0 + 2.0 * nu) + a / sqrt((b_y1 * b_y1 + y2 * y2) + nf_a * nf_a)) *
    (1.0 / tan(b)) + b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + of_a * of_a) + y3)
    + 2.0 * a) * (nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + pf_a * pf_a))) * (0.5 /
    sqrt((b_y1 * b_y1 + y2 * y2) + qf_a * qf_a) * (2.0 * y3 + 4.0 * a) + 1.0)) +
    (1.0 - 2.0 * nu) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + rf_a * rf_a) + y3)
    + 2.0 * a) * ((-0.5 * a / powd_snf((b_y1 * b_y1 + y2 * y2) + sf_a * sf_a,
    1.5) * (2.0 * y3 + 4.0 * a) * (1.0 / tan(b)) - b_y1 / (tf_a * tf_a) * (nu +
    a / sqrt((b_y1 * b_y1 + y2 * y2) + uf_a * uf_a)) * (0.5 / sqrt((b_y1 * b_y1
    + y2 * y2) + vf_a * vf_a) * (2.0 * y3 + 4.0 * a) + 1.0)) - 0.5 * b_y1 /
                  ((sqrt((b_y1 * b_y1 + y2 * y2) + wf_a * wf_a) + y3) + 2.0 * a)
                  * a / powd_snf((b_y1 * b_y1 + y2 * y2) + xf_a * xf_a, 1.5) *
                  (2.0 * y3 + 4.0 * a))) + (1.0 - 2.0 * nu) * y2 * (1.0 / tan(b))
    / (yf_a * yf_a) * (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2) + ag_a * ag_a) /
                       cos(b)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + bg_a *
    bg_a) * (2.0 * y3 + 4.0 * a) + cos(b))) + 0.5 * (1.0 - 2.0 * nu) * y2 * (1.0
    / tan(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + cg_a * cg_a) - b_y1 * sin(b)) +
                 (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 *
    y2) + dg_a * dg_a, 1.5) / cos(b) * (2.0 * y3 + 4.0 * a)) - a / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + eg_a * eg_a, 1.5) * y2 * (1.0 / tan(b))) + 1.5 *
    a * y2 * (y3 + a) * (1.0 / tan(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) +
    fg_a * fg_a, 2.5) * (2.0 * y3 + 4.0 * a)) + y2 / sqrt((b_y1 * b_y1 + y2 * y2)
    + gg_a * gg_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + hg_a * hg_a) + y3) + 2.0 *
                      a) * (((1.0 - 2.0 * nu) * (1.0 / tan(b)) - 2.0 * nu * b_y1
    / ((sqrt((b_y1 * b_y1 + y2 * y2) + ig_a * ig_a) + y3) + 2.0 * a)) - a * b_y1
    / sqrt((b_y1 * b_y1 + y2 * y2) + jg_a * jg_a) * (1.0 / sqrt((b_y1 * b_y1 +
    y2 * y2) + kg_a * kg_a) + 1.0 / ((sqrt((b_y1 * b_y1 + y2 * y2) + lg_a * lg_a)
    + y3) + 2.0 * a)))) - 0.5 * y2 * (y3 + a) / powd_snf((b_y1 * b_y1 + y2 *
    y2) + mg_a * mg_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ng_a * ng_a) + y3)
    + 2.0 * a) * (((1.0 - 2.0 * nu) * (1.0 / tan(b)) - 2.0 * nu * b_y1 / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + og_a * og_a) + y3) + 2.0 * a)) - a * b_y1 / sqrt
                  ((b_y1 * b_y1 + y2 * y2) + pg_a * pg_a) * (1.0 / sqrt((b_y1 *
    b_y1 + y2 * y2) + qg_a * qg_a) + 1.0 / ((sqrt((b_y1 * b_y1 + y2 * y2) + rg_a
    * rg_a) + y3) + 2.0 * a))) * (2.0 * y3 + 4.0 * a)) - y2 * (y3 + a) / sqrt
    ((b_y1 * b_y1 + y2 * y2) + sg_a * sg_a) / (tg_a * tg_a) * (((1.0 - 2.0 * nu)
    * (1.0 / tan(b)) - 2.0 * nu * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + ug_a *
    ug_a) + y3) + 2.0 * a)) - a * b_y1 / sqrt((b_y1 * b_y1 + y2 * y2) + vg_a *
    vg_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + wg_a * wg_a) + 1.0 / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + xg_a * xg_a) + y3) + 2.0 * a))) * (0.5 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + yg_a * yg_a) * (2.0 * y3 + 4.0 * a) + 1.0)) + y2 *
                        (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) + ah_a * ah_a) /
                        ((sqrt((b_y1 * b_y1 + y2 * y2) + bh_a * bh_a) + y3) +
    2.0 * a) * ((2.0 * nu * b_y1 / (ch_a * ch_a) * (0.5 / sqrt((b_y1 * b_y1 + y2
    * y2) + dh_a * dh_a) * (2.0 * y3 + 4.0 * a) + 1.0) + 0.5 * a * b_y1 /
                 powd_snf((b_y1 * b_y1 + y2 * y2) + eh_a * eh_a, 1.5) * (1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + fh_a * fh_a) + 1.0 / ((sqrt((b_y1 * b_y1 + y2
    * y2) + gh_a * gh_a) + y3) + 2.0 * a)) * (2.0 * y3 + 4.0 * a)) - a * b_y1 /
                sqrt((b_y1 * b_y1 + y2 * y2) + hh_a * hh_a) * (-0.5 /
    powd_snf((b_y1 * b_y1 + y2 * y2) + ih_a * ih_a, 1.5) * (2.0 * y3 + 4.0 *
    a) - 1.0 / (jh_a * jh_a) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + kh_a * kh_a)
    * (2.0 * y3 + 4.0 * a) + 1.0)))) + y2 * (1.0 / tan(b)) / sqrt((b_y1 * b_y1 +
    y2 * y2) + lh_a * lh_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + mh_a * mh_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((-2.0 + 2.0 * nu) * cos(b) +
    ((sqrt((b_y1 * b_y1 + y2 * y2) + nh_a * nh_a) * cos(b) + y3) + 2.0 * a) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + oh_a * oh_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)) * (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2) + ph_a * ph_a) / cos
                    (b))) + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) + qh_a
    * qh_a) / cos(b))) - 0.5 * y2 * (y3 + a) * (1.0 / tan(b)) / powd_snf
                      ((b_y1 * b_y1 + y2 * y2) + rh_a * rh_a, 1.5) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + sh_a * sh_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b)) * (((-2.0 + 2.0 * nu) * cos(b) + ((sqrt((b_y1 * b_y1 + y2 * y2) +
    th_a * th_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    uh_a * uh_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (1.0 + a / sqrt
    ((b_y1 * b_y1 + y2 * y2) + vh_a * vh_a) / cos(b))) + a * (y3 + 2.0 * a) /
               ((b_y1 * b_y1 + y2 * y2) + wh_a * wh_a) / cos(b)) * (2.0 * y3 +
    4.0 * a)) - y2 * (y3 + a) * (1.0 / tan(b)) / sqrt((b_y1 * b_y1 + y2 * y2) +
    xh_a * xh_a) / (yh_a * yh_a) * (((-2.0 + 2.0 * nu) * cos(b) + ((sqrt((b_y1 *
    b_y1 + y2 * y2) + ai_a * ai_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + bi_a * bi_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2) + ci_a * ci_a) / cos(b))) + a * (y3
    + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) + di_a * di_a) / cos(b)) * (0.5 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + ei_a * ei_a) * (2.0 * y3 + 4.0 * a) + cos(b))) +
                    y2 * (y3 + a) * (1.0 / tan(b)) / sqrt((b_y1 * b_y1 + y2 * y2)
    + fi_a * fi_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + gi_a * gi_a) - b_y1 * sin
                       (b)) + (y3 + 2.0 * a) * cos(b)) * (((((0.5 / sqrt((b_y1 *
    b_y1 + y2 * y2) + hi_a * hi_a) * cos(b) * (2.0 * y3 + 4.0 * a) + 1.0) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + ii_a * ii_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)) * (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2) + ji_a * ji_a) / cos
                    (b)) - ((sqrt((b_y1 * b_y1 + y2 * y2) + ki_a * ki_a) * cos(b)
    + y3) + 2.0 * a) / (li_a * li_a) * (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2) +
    mi_a * mi_a) / cos(b)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + ni_a * ni_a) *
    (2.0 * y3 + 4.0 * a) + cos(b))) - 0.5 * ((sqrt((b_y1 * b_y1 + y2 * y2) +
    oi_a * oi_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    pi_a * pi_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + qi_a * qi_a, 1.5) / cos(b) * (2.0 * y3 + 4.0 * a))
    + a / ((b_y1 * b_y1 + y2 * y2) + ri_a * ri_a) / cos(b)) - a * (y3 + 2.0 * a)
    / (si_a * si_a) / cos(b) * (2.0 * y3 + 4.0 * a))) / M_PI /
              (1.0 - nu))) + 0.5 * B3 * (0.125 * ((1.0 - 2.0 * nu) * sin(b) *
              ((1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * y3 - cos(b)) /
               ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 *
                cos(b)) + (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + ti_a * ti_a) *
    (2.0 * y3 + 4.0 * a) + cos(b)) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ui_a *
    ui_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) - y2 * y2 * sin(b) *
              (((-1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) /
                 ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3
                  * cos(b)) * y3 - 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3)
                 / (vi_a * vi_a) * (1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3)
    * y3 - cos(b))) - 0.5 / powd_snf((b_y1 * b_y1 + y2 * y2) + wi_a * wi_a,
    1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + xi_a * xi_a) - b_y1 * sin(b)) + (y3
    + 2.0 * a) * cos(b)) * (2.0 * y3 + 4.0 * a)) - 1.0 / sqrt((b_y1 * b_y1 + y2 *
    y2) + yi_a * yi_a) / (aj_a * aj_a) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) +
    bj_a * bj_a) * (2.0 * y3 + 4.0 * a) + cos(b)))) / M_PI / (1.0
              - nu) + 0.25 * (((((((1.0 - 2.0 * nu) * (((((-sin(b) * (0.5 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + cj_a * cj_a) * (2.0 * y3 + 4.0 * a) + cos(b)) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + dj_a * dj_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)) + b_y1 / (ej_a * ej_a) * (1.0 + a / sqrt((b_y1 * b_y1 + y2 * y2)
    + fj_a * fj_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + gj_a * gj_a) * (2.0 *
    y3 + 4.0 * a) + 1.0)) + 0.5 * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + hj_a *
    hj_a) + y3) + 2.0 * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + ij_a *
    ij_a, 1.5) * (2.0 * y3 + 4.0 * a)) + sin(b) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + jj_a * jj_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a /
    sqrt((b_y1 * b_y1 + y2 * y2) + kj_a * kj_a))) - (b_y1 * cos(b) + (y3 + 2.0 *
    a) * sin(b)) / (lj_a * lj_a) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) +
    mj_a * mj_a)) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + nj_a * nj_a) * (2.0 *
    y3 + 4.0 * a) + cos(b))) - 0.5 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + oj_a * oj_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + pj_a * pj_a, 1.5) *
    (2.0 * y3 + 4.0 * a)) + b_y1 / sqrt((b_y1 * b_y1 + y2 * y2) + qj_a * qj_a) *
    (a / ((b_y1 * b_y1 + y2 * y2) + rj_a * rj_a) + 1.0 / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + sj_a * sj_a) + y3) + 2.0 * a))) - 0.5 * b_y1 * (y3 + a) /
    powd_snf((b_y1 * b_y1 + y2 * y2) + tj_a * tj_a, 1.5) * (a / ((b_y1 * b_y1
    + y2 * y2) + uj_a * uj_a) + 1.0 / ((sqrt((b_y1 * b_y1 + y2 * y2) + vj_a *
    vj_a) + y3) + 2.0 * a)) * (2.0 * y3 + 4.0 * a)) + b_y1 * (y3 + a) / sqrt
    ((b_y1 * b_y1 + y2 * y2) + wj_a * wj_a) * (-a / (xj_a * xj_a) * (2.0 * y3 +
    4.0 * a) - 1.0 / (yj_a * yj_a) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) + ak_a *
    ak_a) * (2.0 * y3 + 4.0 * a) + 1.0))) - 1.0 / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + bk_a * bk_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((sin(b) * (cos
    (b) - a / sqrt((b_y1 * b_y1 + y2 * y2) + ck_a * ck_a)) + (b_y1 * cos(b) +
    (y3 + 2.0 * a) * sin(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + dk_a * dk_a) *
    (1.0 + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) + ek_a * ek_a))) - 1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + fk_a * fk_a) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + gk_a * gk_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (y2 * y2 * cos
    (b) * sin(b) - a * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + hk_a * hk_a) * ((sqrt((b_y1 * b_y1 + y2 * y2) + ik_a *
    ik_a) * cos(b) + y3) + 2.0 * a)))) + (y3 + a) / (jk_a * jk_a) * ((sin(b) *
    (cos(b) - a / sqrt((b_y1 * b_y1 + y2 * y2) + kk_a * kk_a)) + (b_y1 * cos(b)
    + (y3 + 2.0 * a) * sin(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + lk_a * lk_a) *
    (1.0 + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) + mk_a * mk_a))) - 1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + nk_a * nk_a) / ((sqrt((b_y1 * b_y1 + y2 * y2)
    + ok_a * ok_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (y2 * y2 * cos
    (b) * sin(b) - a * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + pk_a * pk_a) * ((sqrt((b_y1 * b_y1 + y2 * y2) + qk_a *
    qk_a) * cos(b) + y3) + 2.0 * a))) * (0.5 / sqrt((b_y1 * b_y1 + y2 * y2) +
    rk_a * rk_a) * (2.0 * y3 + 4.0 * a) + cos(b))) - (y3 + a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + sk_a * sk_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
              ((((((0.5 * sin(b) * a / powd_snf((b_y1 * b_y1 + y2 * y2) +
    tk_a * tk_a, 1.5) * (2.0 * y3 + 4.0 * a) + sin(b) / sqrt((b_y1 * b_y1 + y2 *
    y2) + uk_a * uk_a) * (1.0 + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) +
    vk_a * vk_a))) - 0.5 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) /
                   powd_snf((b_y1 * b_y1 + y2 * y2) + wk_a * wk_a, 1.5) *
                   (1.0 + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) + xk_a *
    xk_a)) * (2.0 * y3 + 4.0 * a)) + (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) /
                  sqrt((b_y1 * b_y1 + y2 * y2) + yk_a * yk_a) * (a / ((b_y1 *
    b_y1 + y2 * y2) + al_a * al_a) - a * (y3 + 2.0 * a) / (bl_a * bl_a) * (2.0 *
    y3 + 4.0 * a))) + 0.5 / powd_snf((b_y1 * b_y1 + y2 * y2) + cl_a * cl_a,
    1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + dl_a * dl_a) - b_y1 * sin(b)) + (y3
    + 2.0 * a) * cos(b)) * (y2 * y2 * cos(b) * sin(b) - a * (b_y1 * cos(b) + (y3
    + 2.0 * a) * sin(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + el_a * el_a) * ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + fl_a * fl_a) * cos(b) + y3) + 2.0 * a)) * (2.0 *
    y3 + 4.0 * a)) + 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + gl_a * gl_a) / (hl_a *
    hl_a) * (y2 * y2 * cos(b) * sin(b) - a * (b_y1 * cos(b) + (y3 + 2.0 * a) *
    sin(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + il_a * il_a) * ((sqrt((b_y1 * b_y1
    + y2 * y2) + jl_a * jl_a) * cos(b) + y3) + 2.0 * a)) * (0.5 / sqrt((b_y1 *
    b_y1 + y2 * y2) + kl_a * kl_a) * (2.0 * y3 + 4.0 * a) + cos(b))) - 1.0 /
               sqrt((b_y1 * b_y1 + y2 * y2) + ll_a * ll_a) / ((sqrt((b_y1 * b_y1
    + y2 * y2) + ml_a * ml_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((-a
    * sin(b) / sqrt((b_y1 * b_y1 + y2 * y2) + nl_a * nl_a) * ((sqrt((b_y1 * b_y1
    + y2 * y2) + ol_a * ol_a) * cos(b) + y3) + 2.0 * a) + 0.5 * a * (b_y1 * cos
    (b) + (y3 + 2.0 * a) * sin(b)) / powd_snf((b_y1 * b_y1 + y2 * y2) + pl_a *
    pl_a, 1.5) * ((sqrt((b_y1 * b_y1 + y2 * y2) + ql_a * ql_a) * cos(b) + y3) +
                  2.0 * a) * (2.0 * y3 + 4.0 * a)) - a * (b_y1 * cos(b) + (y3 +
    2.0 * a) * sin(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + rl_a * rl_a) * (0.5 /
    sqrt((b_y1 * b_y1 + y2 * y2) + sl_a * sl_a) * cos(b) * (2.0 * y3 + 4.0 * a)
    + 1.0)))) / M_PI / (1.0 - nu))) + 0.5 * B1 * ((0.125 * ((1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - 1.0 / sqrt((b_y1 * b_y1 + y2 * y2)
    + tl_a * tl_a)) - cos(b) * ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b)
    - y3) / sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) - ((sqrt((b_y1 * b_y1 + y2 *
    y2) + ul_a * ul_a) * cos(b) + y3) + 2.0 * a) / sqrt((b_y1 * b_y1 + y2 * y2)
    + vl_a * vl_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + wl_a * wl_a) - b_y1 * sin
                       (b)) + (y3 + 2.0 * a) * cos(b)))) / M_PI /
             (1.0 - nu) + 0.125 * y2 * ((-1.0 / powd_snf((b_y1 * b_y1 + y2 *
    y2) + y3 * y3, 1.5) * y2 + 1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + xl_a *
    xl_a, 1.5) * y2) - cos(b) * (((((1.0 / ((b_y1 * b_y1 + y2 * y2) + y3 * y3) *
    cos(b) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) -
                   y3 * cos(b)) - (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos
    (b) - y3) / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 * y3, 1.5) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) * y2) -
    (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b) - y3) / ((b_y1 * b_y1 + y2
    * y2) + y3 * y3) / (yl_a * yl_a) * y2) - 1.0 / ((b_y1 * b_y1 + y2 * y2) +
    am_a * am_a) * cos(b) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + bm_a * bm_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) + ((sqrt((b_y1 * b_y1 + y2 * y2)
    + cm_a * cm_a) * cos(b) + y3) + 2.0 * a) / powd_snf((b_y1 * b_y1 + y2 *
    y2) + dm_a * dm_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + em_a * em_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * y2) + ((sqrt((b_y1 * b_y1 + y2 *
    y2) + fm_a * fm_a) * cos(b) + y3) + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) +
    gm_a * gm_a) / (hm_a * hm_a) * y2)) / M_PI / (1.0 - nu)) +
            0.25 * ((((((((2.0 - 2.0 * nu) * (((((((1.0 - 2.0 * nu) * ((-1.0 /
    b_y1 / (1.0 + y2 * y2 / (b_y1 * b_y1)) + 1.0 / (b_y1 * cos(b) + (y3 + 2.0 *
    a) * sin(b)) / (1.0 + y2 * y2 / (im_a * im_a))) + ((sqrt((b_y1 * b_y1 + y2 *
    y2) + jm_a * jm_a) * sin(b) / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin
    (b)) + y2 * y2 * cos(b)) + y2 * y2 / sqrt((b_y1 * b_y1 + y2 * y2) + km_a *
    km_a) * sin(b) / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2
    * cos(b))) - 2.0 * (y2 * y2) * sqrt((b_y1 * b_y1 + y2 * y2) + lm_a * lm_a) *
    sin(b) / (mm_a * mm_a) * cos(b)) / (1.0 + y2 * y2 * ((b_y1 * b_y1 + y2 * y2)
    + nm_a * nm_a) * (p_x * p_x) / (om_a * om_a))) * (1.0 / tan(b)) + 1.0 /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + pm_a * pm_a) + y3) + 2.0 * a) * (2.0 * nu +
    a / sqrt((b_y1 * b_y1 + y2 * y2) + qm_a * qm_a))) - y2 * y2 / (rm_a * rm_a) *
    (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + sm_a * sm_a)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + tm_a * tm_a)) - y2 * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    um_a * um_a) + y3) + 2.0 * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2) +
    vm_a * vm_a, 1.5)) - cos(b) / ((sqrt((b_y1 * b_y1 + y2 * y2) + wm_a * wm_a)
    - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + xm_a * xm_a))) + y2 * y2 * cos(b) / (ym_a * ym_a) * (cos(b)
    + a / sqrt((b_y1 * b_y1 + y2 * y2) + an_a * an_a)) / sqrt((b_y1 * b_y1 + y2 *
    y2) + bn_a * bn_a)) + y2 * y2 * cos(b) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    cn_a * cn_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + dn_a * dn_a, 1.5)) + (y3 + a) / sqrt((b_y1 * b_y1
    + y2 * y2) + en_a * en_a) * (2.0 * nu / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    fn_a * fn_a) + y3) + 2.0 * a) + a / ((b_y1 * b_y1 + y2 * y2) + gn_a * gn_a)))
    - y2 * y2 * (y3 + a) / powd_snf((b_y1 * b_y1 + y2 * y2) + hn_a * hn_a,
    1.5) * (2.0 * nu / ((sqrt((b_y1 * b_y1 + y2 * y2) + in_a * in_a) + y3) + 2.0
                        * a) + a / ((b_y1 * b_y1 + y2 * y2) + jn_a * jn_a))) +
                        y2 * (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) + kn_a *
    kn_a) * (-2.0 * nu / (ln_a * ln_a) / sqrt((b_y1 * b_y1 + y2 * y2) + mn_a *
    mn_a) * y2 - 2.0 * a / (nn_a * nn_a) * y2)) + (y3 + a) * cos(b) / sqrt((b_y1
    * b_y1 + y2 * y2) + on_a * on_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + pn_a *
    pn_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((1.0 - 2.0 * nu) -
    ((sqrt((b_y1 * b_y1 + y2 * y2) + qn_a * qn_a) * cos(b) + y3) + 2.0 * a) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + rn_a * rn_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + sn_a * sn_a))) -
    a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) + tn_a * tn_a))) - y2 * y2 *
                      (y3 + a) * cos(b) / powd_snf((b_y1 * b_y1 + y2 * y2) +
    un_a * un_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + vn_a * vn_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((1.0 - 2.0 * nu) - ((sqrt((b_y1 *
    b_y1 + y2 * y2) + wn_a * wn_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + xn_a * xn_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + yn_a * yn_a))) - a * (y3 + 2.0 *
    a) / ((b_y1 * b_y1 + y2 * y2) + ao_a * ao_a))) - y2 * y2 * (y3 + a) * cos(b)
                     / ((b_y1 * b_y1 + y2 * y2) + bo_a * bo_a) / (co_a * co_a) *
                     (((1.0 - 2.0 * nu) - ((sqrt((b_y1 * b_y1 + y2 * y2) + do_a *
    do_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + eo_a *
    eo_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1
    * b_y1 + y2 * y2) + fo_a * fo_a))) - a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2
    * y2) + go_a * go_a))) + y2 * (y3 + a) * cos(b) / sqrt((b_y1 * b_y1 + y2 *
    y2) + ho_a * ho_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + io_a * io_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((-1.0 / sqrt((b_y1 * b_y1 + y2 * y2)
    + jo_a * jo_a) * cos(b) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + ko_a * ko_a)
    - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + lo_a * lo_a)) + ((sqrt((b_y1 * b_y1 + y2 * y2) + mo_a *
    mo_a) * cos(b) + y3) + 2.0 * a) / (no_a * no_a) * (cos(b) + a / sqrt((b_y1 *
    b_y1 + y2 * y2) + oo_a * oo_a)) / sqrt((b_y1 * b_y1 + y2 * y2) + po_a * po_a)
    * y2) + ((sqrt((b_y1 * b_y1 + y2 * y2) + qo_a * qo_a) * cos(b) + y3) + 2.0 *
             a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + ro_a * ro_a) - b_y1 * sin(b))
                   + (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 * b_y1 +
    y2 * y2) + so_a * so_a, 1.5) * y2) + 2.0 * a * (y3 + 2.0 * a) / (to_a * to_a)
              * y2)) / M_PI / (1.0 - nu))) + 0.5 * B2 * (0.125 *
           ((((((((-1.0 + 2.0 * nu) * sin(b) * (1.0 / sqrt((b_y1 * b_y1 + y2 *
    y2) + y3 * y3) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin
    (b)) - y3 * cos(b)) - 1.0 / sqrt((b_y1 * b_y1 + y2 * y2) + uo_a * uo_a) * y2
    / ((sqrt((b_y1 * b_y1 + y2 * y2) + vo_a * vo_a) - b_y1 * sin(b)) + (y3 + 2.0
    * a) * cos(b))) - b_y1 * (-1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + y3 *
    y3, 1.5) * y2 + 1.0 / powd_snf((b_y1 * b_y1 + y2 * y2) + wo_a * wo_a, 1.5)
    * y2)) + (b_y1 * cos(b) - y3 * sin(b)) / ((b_y1 * b_y1 + y2 * y2) + y3 * y3)
                 * cos(b) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) -
    b_y1 * sin(b)) - y3 * cos(b))) - (b_y1 * cos(b) - y3 * sin(b)) * (sqrt((b_y1
    * b_y1 + y2 * y2) + y3 * y3) * cos(b) - y3) / powd_snf((b_y1 * b_y1 + y2 *
    y2) + y3 * y3, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin
    (b)) - y3 * cos(b)) * y2) - (b_y1 * cos(b) - y3 * sin(b)) * (sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) * cos(b) - y3) / ((b_y1 * b_y1 + y2 * y2) + y3 *
    y3) / (xo_a * xo_a) * y2) - (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) /
              ((b_y1 * b_y1 + y2 * y2) + yo_a * yo_a) * cos(b) * y2 / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + ap_a * ap_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b))) + (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * ((sqrt((b_y1 * b_y1 +
    y2 * y2) + bp_a * bp_a) * cos(b) + y3) + 2.0 * a) / powd_snf((b_y1 * b_y1
    + y2 * y2) + cp_a * cp_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + dp_a *
    dp_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * y2) + (b_y1 * cos(b) +
             (y3 + 2.0 * a) * sin(b)) * ((sqrt((b_y1 * b_y1 + y2 * y2) + ep_a *
    ep_a) * cos(b) + y3) + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) + fp_a * fp_a) /
            (gp_a * gp_a) * y2) / M_PI / (1.0 - nu) + 0.25 *
           (((((((((-2.0 + 2.0 * nu) * (1.0 - 2.0 * nu) * (1.0 / tan(b)) * (1.0 /
    sqrt((b_y1 * b_y1 + y2 * y2) + hp_a * hp_a) * y2 / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + ip_a * ip_a) + y3) + 2.0 * a) - cos(b) / sqrt((b_y1 * b_y1 + y2 * y2)
    + jp_a * jp_a) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + kp_a * kp_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b))) + (2.0 - 2.0 * nu) * b_y1 / (lp_a * lp_a)
                   * (2.0 * nu + a / sqrt((b_y1 * b_y1 + y2 * y2) + mp_a * mp_a))
                   / sqrt((b_y1 * b_y1 + y2 * y2) + np_a * np_a) * y2) + (2.0 -
    2.0 * nu) * b_y1 / ((sqrt((b_y1 * b_y1 + y2 * y2) + op_a * op_a) + y3) + 2.0
                        * a) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + pp_a *
    pp_a, 1.5) * y2) - (2.0 - 2.0 * nu) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin
    (b)) / (qp_a * qp_a) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + rp_a *
    rp_a)) / sqrt((b_y1 * b_y1 + y2 * y2) + sp_a * sp_a) * y2) - (2.0 - 2.0 * nu)
                * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + tp_a * tp_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
                a / powd_snf((b_y1 * b_y1 + y2 * y2) + up_a * up_a, 1.5) * y2)
               - (y3 + a) / powd_snf((b_y1 * b_y1 + y2 * y2) + vp_a * vp_a,
    1.5) * (((1.0 - 2.0 * nu) * (1.0 / tan(b)) - 2.0 * nu * b_y1 / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + wp_a * wp_a) + y3) + 2.0 * a)) - a * b_y1 / ((b_y1 * b_y1
    + y2 * y2) + xp_a * xp_a)) * y2) + (y3 + a) / sqrt((b_y1 * b_y1 + y2 * y2) +
    yp_a * yp_a) * (2.0 * nu * b_y1 / (aq_a * aq_a) / sqrt((b_y1 * b_y1 + y2 *
    y2) + bq_a * bq_a) * y2 + 2.0 * a * b_y1 / (cq_a * cq_a) * y2)) + (y3 + a) /
             (dq_a * dq_a) * ((cos(b) * sin(b) + ((sqrt((b_y1 * b_y1 + y2 * y2)
    + eq_a * eq_a) * cos(b) + y3) + 2.0 * a) * (1.0 / tan(b)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + fq_a * fq_a) * ((2.0 - 2.0 * nu) * cos(b) - ((sqrt((b_y1 *
    b_y1 + y2 * y2) + gq_a * gq_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + hq_a * hq_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))))
              + a / sqrt((b_y1 * b_y1 + y2 * y2) + iq_a * iq_a) * ((sin(b) - (y3
    + 2.0 * a) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / ((b_y1 * b_y1 + y2 *
    y2) + jq_a * jq_a)) - (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) * ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + kq_a * kq_a) * cos(b) + y3) + 2.0 * a) / sqrt
    ((b_y1 * b_y1 + y2 * y2) + lq_a * lq_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) +
    mq_a * mq_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)))) / sqrt((b_y1 *
    b_y1 + y2 * y2) + nq_a * nq_a) * y2) - (y3 + a) / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + oq_a * oq_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((((1.0 /
    ((b_y1 * b_y1 + y2 * y2) + pq_a * pq_a) * cos(b) * y2 * (1.0 / tan(b)) *
    ((2.0 - 2.0 * nu) * cos(b) - ((sqrt((b_y1 * b_y1 + y2 * y2) + qq_a * qq_a) *
    cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + rq_a * rq_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) - ((sqrt((b_y1 * b_y1 + y2 * y2)
    + sq_a * sq_a) * cos(b) + y3) + 2.0 * a) * (1.0 / tan(b)) / powd_snf
    ((b_y1 * b_y1 + y2 * y2) + tq_a * tq_a, 1.5) * ((2.0 - 2.0 * nu) * cos(b) -
    ((sqrt((b_y1 * b_y1 + y2 * y2) + uq_a * uq_a) * cos(b) + y3) + 2.0 * a) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + vq_a * vq_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b))) * y2) + ((sqrt((b_y1 * b_y1 + y2 * y2) + wq_a * wq_a) * cos(b)
    + y3) + 2.0 * a) * (1.0 / tan(b)) / sqrt((b_y1 * b_y1 + y2 * y2) + xq_a *
    xq_a) * (-cos(b) / sqrt((b_y1 * b_y1 + y2 * y2) + yq_a * yq_a) * y2 / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + ar_a * ar_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b)) + ((sqrt((b_y1 * b_y1 + y2 * y2) + br_a * br_a) * cos(b) + y3) + 2.0
               * a) / (cr_a * cr_a) / sqrt((b_y1 * b_y1 + y2 * y2) + dr_a * dr_a)
             * y2)) - a / powd_snf((b_y1 * b_y1 + y2 * y2) + er_a * er_a, 1.5)
              * ((sin(b) - (y3 + 2.0 * a) * (b_y1 * cos(b) + (y3 + 2.0 * a) *
    sin(b)) / ((b_y1 * b_y1 + y2 * y2) + fr_a * fr_a)) - (b_y1 * cos(b) + (y3 +
    2.0 * a) * sin(b)) * ((sqrt((b_y1 * b_y1 + y2 * y2) + gr_a * gr_a) * cos(b)
    + y3) + 2.0 * a) / sqrt((b_y1 * b_y1 + y2 * y2) + hr_a * hr_a) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + ir_a * ir_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b))) * y2) + a / sqrt((b_y1 * b_y1 + y2 * y2) + jr_a * jr_a) * (((2.0 *
    (y3 + 2.0 * a) * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / (kr_a * kr_a) *
    y2 - (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) / ((b_y1 * b_y1 + y2 * y2) +
    lr_a * lr_a) * cos(b) * y2 / ((sqrt((b_y1 * b_y1 + y2 * y2) + mr_a * mr_a) -
    b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b))) + (b_y1 * cos(b) + (y3 + 2.0 * a)
    * sin(b)) * ((sqrt((b_y1 * b_y1 + y2 * y2) + nr_a * nr_a) * cos(b) + y3) +
                 2.0 * a) / powd_snf((b_y1 * b_y1 + y2 * y2) + or_a * or_a,
    1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + pr_a * pr_a) - b_y1 * sin(b)) + (y3
    + 2.0 * a) * cos(b)) * y2) + (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) *
              ((sqrt((b_y1 * b_y1 + y2 * y2) + qr_a * qr_a) * cos(b) + y3) + 2.0
               * a) / ((b_y1 * b_y1 + y2 * y2) + rr_a * rr_a) / (sr_a * sr_a) *
              y2))) / M_PI / (1.0 - nu))) + 0.5 * B3 * (0.125 *
    (((2.0 - 2.0 * nu) * (((1.0 / (b_y1 * cos(b) - y3 * sin(b)) / (1.0 + y2 * y2
    / (tr_a * tr_a)) + ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) /
    (b_y1 * (b_y1 * cos(b) - y3 * sin(b)) + y2 * y2 * cos(b)) + y2 * y2 / sqrt
    ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * sin(b) / (b_y1 * (b_y1 * cos(b) - y3 *
    sin(b)) + y2 * y2 * cos(b))) - 2.0 * (y2 * y2) * sqrt((b_y1 * b_y1 + y2 * y2)
    + y3 * y3) * sin(b) / (ur_a * ur_a) * cos(b)) / (1.0 + y2 * y2 * ((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) * (lu_a * lu_a) / (vr_a * vr_a))) - 1.0 / (b_y1 *
    cos(b) + (y3 + 2.0 * a) * sin(b)) / (1.0 + y2 * y2 / (wr_a * wr_a))) -
    ((sqrt((b_y1 * b_y1 + y2 * y2) + xr_a * xr_a) * sin(b) / (b_y1 * (b_y1 * cos
    (b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2 * cos(b)) + y2 * y2 / sqrt((b_y1 *
    b_y1 + y2 * y2) + yr_a * yr_a) * sin(b) / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0
    * a) * sin(b)) + y2 * y2 * cos(b))) - 2.0 * (y2 * y2) * sqrt((b_y1 * b_y1 +
    y2 * y2) + as_a * as_a) * sin(b) / (bs_a * bs_a) * cos(b)) / (1.0 + y2 * y2 *
    ((b_y1 * b_y1 + y2 * y2) + cs_a * cs_a) * (mu_a * mu_a) / (ds_a * ds_a))) +
      sin(b) * ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b) - y3) / sqrt
                ((b_y1 * b_y1 + y2 * y2) + y3 * y3) / ((sqrt((b_y1 * b_y1 + y2 *
    y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) - ((sqrt((b_y1 * b_y1 + y2 *
    y2) + es_a * es_a) * cos(b) + y3) + 2.0 * a) / sqrt((b_y1 * b_y1 + y2 * y2)
    + fs_a * fs_a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + gs_a * gs_a) - b_y1 * sin
                       (b)) + (y3 + 2.0 * a) * cos(b)))) + y2 * sin(b) *
     (((((1.0 / ((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b) * y2 / ((sqrt((b_y1
    * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin(b)) - y3 * cos(b)) - (sqrt((b_y1 *
    b_y1 + y2 * y2) + y3 * y3) * cos(b) - y3) / powd_snf((b_y1 * b_y1 + y2 *
    y2) + y3 * y3, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) - b_y1 * sin
    (b)) - y3 * cos(b)) * y2) - (sqrt((b_y1 * b_y1 + y2 * y2) + y3 * y3) * cos(b)
    - y3) / ((b_y1 * b_y1 + y2 * y2) + y3 * y3) / (hs_a * hs_a) * y2) - 1.0 /
        ((b_y1 * b_y1 + y2 * y2) + is_a * is_a) * cos(b) * y2 / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + js_a * js_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)))
       + ((sqrt((b_y1 * b_y1 + y2 * y2) + ks_a * ks_a) * cos(b) + y3) + 2.0 * a)
       / powd_snf((b_y1 * b_y1 + y2 * y2) + ls_a * ls_a, 1.5) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + ms_a * ms_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
       y2) + ((sqrt((b_y1 * b_y1 + y2 * y2) + ns_a * ns_a) * cos(b) + y3) + 2.0 *
              a) / ((b_y1 * b_y1 + y2 * y2) + os_a * os_a) / (ps_a * ps_a) * y2))
    / M_PI / (1.0 - nu) + 0.25 * ((((((((2.0 - 2.0 * nu) * ((-1.0 /
    b_y1 / (1.0 + y2 * y2 / (b_y1 * b_y1)) + 1.0 / (b_y1 * cos(b) + (y3 + 2.0 *
    a) * sin(b)) / (1.0 + y2 * y2 / (qs_a * qs_a))) + ((sqrt((b_y1 * b_y1 + y2 *
    y2) + rs_a * rs_a) * sin(b) / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin
    (b)) + y2 * y2 * cos(b)) + y2 * y2 / sqrt((b_y1 * b_y1 + y2 * y2) + ss_a *
    ss_a) * sin(b) / (b_y1 * (b_y1 * cos(b) + (y3 + 2.0 * a) * sin(b)) + y2 * y2
                      * cos(b))) - 2.0 * (y2 * y2) * sqrt((b_y1 * b_y1 + y2 * y2)
    + ts_a * ts_a) * sin(b) / (us_a * us_a) * cos(b)) / (1.0 + y2 * y2 * ((b_y1 *
    b_y1 + y2 * y2) + vs_a * vs_a) * (nu_a * nu_a) / (ws_a * ws_a))) + (2.0 -
    2.0 * nu) * sin(b) / ((sqrt((b_y1 * b_y1 + y2 * y2) + xs_a * xs_a) - b_y1 *
    sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 *
    y2) + ys_a * ys_a))) - (2.0 - 2.0 * nu) * (y2 * y2) * sin(b) / (at_a * at_a)
    * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + bt_a * bt_a)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + ct_a * ct_a)) - (2.0 - 2.0 * nu) * (y2 * y2) * sin(b) /
    ((sqrt((b_y1 * b_y1 + y2 * y2) + dt_a * dt_a) - b_y1 * sin(b)) + (y3 + 2.0 *
    a) * cos(b)) * a / powd_snf((b_y1 * b_y1 + y2 * y2) + et_a * et_a, 1.5))
    + (y3 + a) * sin(b) / sqrt((b_y1 * b_y1 + y2 * y2) + ft_a * ft_a) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + gt_a * gt_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b)) * ((1.0 + ((sqrt((b_y1 * b_y1 + y2 * y2) + ht_a * ht_a) * cos(b) +
                        y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + it_a *
    it_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (cos(b) + a / sqrt((b_y1
    * b_y1 + y2 * y2) + jt_a * jt_a))) + a * (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2
    * y2) + kt_a * kt_a))) - y2 * y2 * (y3 + a) * sin(b) / powd_snf((b_y1 *
    b_y1 + y2 * y2) + lt_a * lt_a, 1.5) / ((sqrt((b_y1 * b_y1 + y2 * y2) + mt_a *
    mt_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * ((1.0 + ((sqrt((b_y1 *
    b_y1 + y2 * y2) + nt_a * nt_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + ot_a * ot_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + pt_a * pt_a))) + a * (y3 + 2.0 *
    a) / ((b_y1 * b_y1 + y2 * y2) + qt_a * qt_a))) - y2 * y2 * (y3 + a) * sin(b)
    / ((b_y1 * b_y1 + y2 * y2) + rt_a * rt_a) / (st_a * st_a) * ((1.0 + ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + tt_a * tt_a) * cos(b) + y3) + 2.0 * a) / ((sqrt
    ((b_y1 * b_y1 + y2 * y2) + ut_a * ut_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) *
    cos(b)) * (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + vt_a * vt_a))) + a *
    (y3 + 2.0 * a) / ((b_y1 * b_y1 + y2 * y2) + wt_a * wt_a))) + y2 * (y3 + a) *
    sin(b) / sqrt((b_y1 * b_y1 + y2 * y2) + xt_a * xt_a) / ((sqrt((b_y1 * b_y1 +
    y2 * y2) + yt_a * yt_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * (((1.0
    / sqrt((b_y1 * b_y1 + y2 * y2) + au_a * au_a) * cos(b) * y2 / ((sqrt((b_y1 *
    b_y1 + y2 * y2) + bu_a * bu_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) *
    (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + cu_a * cu_a)) - ((sqrt((b_y1 *
    b_y1 + y2 * y2) + du_a * du_a) * cos(b) + y3) + 2.0 * a) / (eu_a * eu_a) *
    (cos(b) + a / sqrt((b_y1 * b_y1 + y2 * y2) + fu_a * fu_a)) / sqrt((b_y1 *
    b_y1 + y2 * y2) + gu_a * gu_a) * y2) - ((sqrt((b_y1 * b_y1 + y2 * y2) + hu_a
    * hu_a) * cos(b) + y3) + 2.0 * a) / ((sqrt((b_y1 * b_y1 + y2 * y2) + iu_a *
    iu_a) - b_y1 * sin(b)) + (y3 + 2.0 * a) * cos(b)) * a / powd_snf((b_y1 *
    b_y1 + y2 * y2) + ju_a * ju_a, 1.5) * y2) - 2.0 * a * (y3 + 2.0 * a) / (ku_a
    * ku_a) * y2)) / M_PI / (1.0 - nu));
}


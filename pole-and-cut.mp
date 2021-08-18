input mycurves;
outputtemplate := "%j-%c.mps";

def draw_pole(expr p, R) =
  begingroup;
    pair pa, pb, pc, pd;
    pa := p shifted ((R, 0) rotated 45);
    pb := p shifted ((R, 0) rotated 135);
    pc := p shifted ((R, 0) rotated 225);
    pd := p shifted ((R, 0) rotated 315);
    draw pa -- pc;
    draw pb -- pd;
  endgroup;
enddef;

def draw_cut(expr cut_a, cut_b, h) =
  begingroup;
    pair pa, pb, pc, pd;
    pair off;
    cut_angle := angle(cut_b - cut_a);
    off := ((0, h/2) rotated cut_angle);
    pa := cut_b shifted off;
    pb := cut_a shifted off;
    pc := cut_a shifted -off;
    pd := cut_b shifted -off;
    path cut_cap, cut_circle;
    cut_circle := fullcircle scaled h
      rotated (90 + cut_angle) shifted cut_a;
    cut_cap := subpath (0.0, 0.5*length(cut_circle)) of cut_circle;
    %draw pa -- pb -- cut_circle -- pc -- pd;
    path cut_fill;
    cut_fill := buildcycle( pa -- pb -- cut_cap -- pc -- pd -- pa);
    pickup pencircle scaled 1pt;
    %fill cut_fill withcolor 0.7white;
    picture cut_dash;
    path dash_clip;
    dash_clip := buildcycle(pa -- pb -- pc -- pd -- pa);
    n := abs(cut_b - cut_a) / h;
    pair d_a, d_b, d_off;
    d_a := cut_a shifted ( (h / sqrt(2.0), 0) rotated (45 + cut_angle) );
    d_b := 2 * cut_a - d_a;
    d_off := (h, 0)  rotated cut_angle;
    pickup pencircle scaled 1pt;
    cut_dash := image(
	for i = 0 upto n:
	    draw (d_a -- d_b) shifted (d_off * i);
	endfor
    );
    fill dash_clip withcolor 0.8white;
    clip cut_dash to dash_clip;
    draw cut_dash;
    draw pa -- pb -- cut_cap -- pc -- pd;
    fill cut_circle withcolor white;
      draw cut_circle ;


  endgroup;
enddef;

def draw_pole_contour(expr pole, R) =
 begingroup;
  path pth_pole;
  ahlength := 0.2u;
  pth_pole := fullcircle rotated 225 scaled (R) shifted pole;
  pth_pole := subpath((0.02, 0.98)*length(pth_pole)) of pth_pole;
  pickup pencircle scaled 1.2pt;  
  drawpatharr pth_pole withcolor red;
 endgroup;
enddef;

def draw_cut_contour(expr cut_a, cut_b, R) =
  begingroup;
    pair pa, pb, pc, pd, pab, paba, pabb,  pcd, pcdc, pcdd;
    pair off;
    cut_angle := angle(cut_b - cut_a);
    off := ((0, R) rotated cut_angle);
    pa := cut_b shifted off;
    pb := cut_a shifted off;
    pc := cut_a shifted -off;
    pd := cut_b shifted -off;
    path cut_cap, cut_circle;
    cut_circle := fullcircle scaled (2*R)
      rotated (90 + cut_angle) shifted cut_a;
    cut_cap := subpath (0.0, 0.5*length(cut_circle)) of cut_circle;
    path pth_cut;
    pab := 0.5 *( pa + pb);
    paba := pab shifted (0.02 * (pa - pb));
    pabb := 2 * pab - paba;
    pcd := 0.5 * (pc + pd);
    pcdc := pcd shifted (0.02 * (pc - pd));
    pickup pencircle scaled 1.2pt; 
    %pth_cut := pa -- pb -- cut_cap -- pc -- pd;
    drawpatharr pa -- paba withcolor dark_green;
    drawpatharr pab -- pb -- cut_cap -- pcdc withcolor dark_green;
    draw pcd -- pd withcolor dark_green;
  endgroup;
enddef;

beginfig(1);
  u := 13mm;
  pair o, ox, oxx, oy, oyy;
  o := (3u, 3u);
  ox := o shifted (2u, 0);
  oxx := 2 * o - ox;
  oy := o shifted (0, 3.0u);
  oyy := o shifted (0.0, -2.5u);

  ahlength := 0.2u;
  pickup pencircle scaled 1.5pt;
  drawpatharr oxx--ox;
  drawpatharr oyy--oy;

  pair pole_u, pole_d, cut_ua, cut_ub,  cut_da, cut_db;
  pole_u := o shifted (0, 0.7u);
  pole_d = 2 * o - pole_u;
  R_pole = 0.1u;
  h_cut = 0.1u;
  cut_ua := o shifted (0, 1.5u);
  cut_ub := cut_ua shifted (0, u);
  cut_da := 2*o - cut_ua;
  cut_db := 2*o - cut_ub;
  pickup pencircle scaled 1.5pt;
  draw_pole(pole_u, R_pole);
  draw_pole(pole_d, R_pole);
  draw_cut(cut_ua, cut_ub, h_cut);
  draw_cut(cut_da, cut_db, h_cut);
  pair c_a, c_b;
  c_a := o shifted (-1.7u, 0.1u);
  c_b := o shifted (1.7u, 0.1u);
  pickup pencircle scaled 1.2pt;
  drawpatharr c_a -- c_b withcolor blue ;

  draw_pole_contour(pole_u, 4*R_pole);
  draw_cut_contour (cut_ua, cut_ub, 2*R_pole);

   label.urt(btex $\mathop{\rm Im}\nolimits q$ etex, oy);
   label.urt(btex $\mathop{\rm Re}\nolimits q$ etex, ox);
   label.rt(btex $-i|k|$ etex, pole_d shifted (1.3*R_pole, 0));
   label.rt(btex $ i|k|$ etex, pole_u shifted (2.2*R_pole, 0));
   label.rt(btex $-i \kappa$ etex, cut_da shifted (R_pole, 0));
   label.rt(btex $i \kappa$ etex, cut_ua shifted (2.2R_pole, 0));

endfig;
end;

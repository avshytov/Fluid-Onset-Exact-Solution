input mycurves;
outputtemplate := "%j-%c.mps";

def draw_particle_path(suffix pta)(expr n) =
   begingroup;
      angle_prev := angle(pta[1] - pta[0]);
      f_prev := 1.0;
      for i = 1 upto n:
         pair pa, pb;
         pa := pta[i - 1];
         pb := pta[i];
         if (ypart (pa) > ypart(o)):
             da  := angle(pb - pa) - angle_prev;
             f := (1.0 + 2.0 * cosd(da)) *  f_prev;
             f_prev := f;
         fi;
         color col;
         if f > 0:
            col := red;
         else:
            col := blue;
         fi;
         R_c := 2pt;
         R_p := 0.25u;
         pair dir_ab;
         dir_ab := (R_c, 0) rotated angle(pb - pa);
         pickup pencircle scaled 0.7pt;
         pair pda, pdb;
         pda := pa + dir_ab;
         pdb := pb - dir_ab;
         if ypart(pb) <= ypart(o):
            pdb := pb;
         fi;
         if ypart(pa) <= ypart(o):
            pda := pa;
          fi;
         %if ypart(pb) > ypart(o) and ypart(pa) > ypart(o):
         drawpatharr (pda)--(pdb) withcolor col;
         %else:
         %
         %fi;
         if (i < n) and (ypart(pb) > ypart(o)):
             fill fullcircle scaled R_c  shifted pb withcolor 0.5white;
             pair ppa, ppb;
             ppb := pb shifted dir_ab;
             ppa := ppb shifted ((R_p/R_c) * dir_ab);
             pickup pencircle scaled 0.5pt;
             drawpatharr ppa -- ppb dashed evenly scaled 0.5;
             pair pc, ppc, ppd, dir_bc;
             pc := pta[i + 1];
             dir_bc := (R_c, 0) rotated angle(pc - pb);
             ppc := pb shifted (-dir_bc);
             ppd := ppc shifted (-(R_p/R_c)*dir_bc);
             drawpatharr ppc -- ppd dashed evenly scaled 0.5;
         fi;
         angle_prev := angle(pb - pa);
      endfor;
   endgroup;
enddef;

def draw_src(expr src, R_src, Nc, col_src, col_bg) =
 begingroup;
   for i = 1 upto Nc - 1:
        color col;
        w := i / (Nc + 0.0);
        w := w * w;
        col := (1 - w) * col_bg + col_src;
         R_i := R_src * (1 - w);
        fill fullcircle scaled R_i shifted src withcolor col;
   endfor;
 endgroup;
enddef;

def draw_continuation(expr p_a, p_b) =
   begingroup;
      pair p_cont;
      p_cont := p_b shifted ((0.7u, 0) rotated angle(p_b - p_a));
      drawpatharr p_b --  p_cont dashed evenly scaled 0.7 withcolor dark_green;
   endgroup;
enddef;

beginfig(1);
  u := 15mm;
  ahlength := 0.12u;
  pair o;
  o := (3u, u);
  pair wa, wb, wc, wd;
  wa := o shifted (3u, 0);
  wb := 2 * o - wa;
  wc := wb shifted (0, -0.2u);
  wd := wc shifted (wa - wb);
  path pth_wall;
  path pth_broken;
  Nb := 500;
  pair pt_broken[];
  pt_broken[0] := wa;
  h_dev := 1.0 * abs(wb - wa) / Nb;
  Np := 0; 
  v_dev := 0.5*h_dev; 
  for i = 1 upto Nb:
    if xpart(pt_broken[Np]) > xpart(wb) - h_dev:
     pair dp;
     %dp := (-uniformdeviate(h_dev), v_dev*(2*uniformdeviate(1.0) - 1.0)); 
     %dp := (-h_dev, v_dev*(2*uniformdeviate(1.0) - 1.0)); 
     dp := (-h_dev, v_dev*normaldeviate); 
     %pt_broken[i] := pt_broken[i - 1] + dp;  
     pt_broken[Np + 1] := (xpart(pt_broken[Np]) - h_dev, ypart(wb) + v_dev * normaldeviate); %pt + dp;  
     Np := Np + 1;
   fi; 
  endfor;
  pth_broken := pt_broken[0]
    for i = 1 upto Np:
    -- pt_broken[i]
    endfor; 
  Nb := i; 
  %pth_wall := buildcycle(wa -- wb -- wc -- wd -- wa);
  pth_wall := buildcycle(wa -- pth_broken --  wb -- wc -- wd -- wa);
  fill pth_wall withcolor 0.8white;
  draw pth_broken withcolor black; 
  pair src;
  src := o shifted (0, 0.7u);


  pair oo, ox, oy;
  oo := o shifted (-2.5u, 0.1u);
  ox := oo shifted (0.5u, 0);
  oy := oo shifted (0, 0.5u);
  drawpatharr oo -- ox;
  drawpatharr oo -- oy;

  %pair pa, paa, pab, pac,  pb, pc, pd;
  %pa := src shifted (0.3u, 0.5u);
  %paa := pa shifted (-0.7u, -0.2u);
  %pab := paa shifted(0.4u, 0.3u);
  %pac := pab shifted()
  %drawpatharr src -- pa;
  %drawpatharr pa  -- paa withcolor red;
  %drawpatharr paa -- pab withcolor red;
  %drawpatharr pab  -- pac withcolor red;
  %path pth_a;
  %pth_a := () scaled u shifted src;
  %draw_particle_path(pth_a);
  pair pt_a[], pt_b[];
  pt_a[0] := src;
  pt_a[1] := pt_a[0] shifted (-0.2u, 0.8u);
  pt_a[2] := pt_a[1] shifted (-0.8u, -1.0u);
  pt_a[3] := pt_a[2] shifted (0.3u, -(ypart(pt_a[2] - o)));
  pt_a[4] := pt_a[3] shifted (0.5u, 0.1u);
  draw_particle_path(pt_a, 4);
  pt_b[0] := src;
  pt_b[1] := pt_b[0] shifted (0.7u, 0.6u);
  pt_b[2] := pt_b[1] shifted (1.2u, -0.6u);
  pt_b[3] := pt_b[2] shifted (-0.2u, -ypart(pt_b[2] - o));
  pt_b[4] := pt_b[3] shifted (-0.4u, 0.25u);
  draw_particle_path(pt_b, 4);

  pb := o shifted (0.2u, 0);
  drawpatharr src -- pb withcolor red;
  pair pbb;
  pbb := pb shifted (pb - src);
  %drawpatharr pb -- pbb withcolor dark_green dashed evenly scaled 0.7;
  draw_continuation(src, pb);
  pair pbc;
  pbc := pb shifted (0.5u, 0.25u);
  drawpatharr pb -- pbc withcolor red;

  draw_continuation(pt_a[2], pt_a[3]);
  draw_continuation(pt_b[2], pt_b[3]);

  pair p_ref;
  p_ref := src shifted (0.7u, 0);
  pickup pencircle scaled 0.3pt;
  draw src -- p_ref;
  R_ref := 0.7*abs(p_ref - src);
  path c_ref;
  c_ref := fullcircle scaled R_ref shifted src;
  c_ref := subpath((0.0, angle(pt_b[1] - src))/360.0 * length(c_ref)) of c_ref;
  draw c_ref;

  R_src := 0.2u;
  Nc := 10;
  draw_src(src, R_src, Nc, red, white);

  label.rt(btex  $x$ etex, ox);
  label.top(btex $y$ etex, oy);
  label.lrt(btex $f^{-}({\bf r}, \alpha)$ etex, pt_b[3] shifted(0.01u, -0.25u));
  label.urt(btex $\alpha$ etex, point (0.3*length(c_ref)) of c_ref);
  label.rt(btex $f_s(x)$ etex, pbc shifted (0.01u, 0.01u));
  label.urt(btex $f^{+}({\bf r}, \alpha)$ etex, (3*pt_b[1] + 1*pt_b[2])/4.0);
  label.llft(btex $J_{\rm ext}$ etex, src);

  pair probe_a, probe_b, probe_c, probe_d;
  path pth_probe ; 
  x_p := xpart(pt_a[3]);
  w_p := 0.1u; 
  pair t_a, t_b;
  t_a := pth_broken intersectiontimes
      (((0, 10u)--(0, -10u)) shifted (x_p + w_p/2, ypart(pt_a[3])));   
  t_b := pth_broken intersectiontimes
      (((0, -10u)--(0, 10u)) shifted (x_p - w_p/2, ypart(pt_a[3])));
  probe_a := point xpart(t_a) of pth_broken;
  probe_b := point xpart(t_b) of pth_broken;
  probe_c := (xpart(probe_b), ypart(wc));
  probe_d := (xpart(probe_a), ypart(wc));
    pth_probe := buildcycle(probe_a
	-- ((subpath (xpart(t_a), xpart(t_b)) of pth_broken))
	-- probe_b -- probe_c -- probe_d -- probe_a);
  fill pth_probe withcolor magenta;
  label.bot(btex $P$ etex, (probe_c + probe_d)/2);   
    
    
endfig;
end;

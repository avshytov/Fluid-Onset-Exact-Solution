verbatimtex
   %\magnification=1440
   \documentclass[12pt]{article}
   \usepackage{amsfonts,bm,color}
   %\usepackage{bm}
   %\usepackage{color}
   \definecolor{darkgreen}{rgb}{0.0,0.8,0.0}
   \begin{document}
etex   

arrowasp := 0.8; 
arrdebug := 0; 

color pale_yellow; 
pale_yellow := (1.0, 1.0, 204/255.0);
color dark_green; 
dark_green := 0.6green; 
color orange; 
orange := red + green*(102.0/255.0) + blue*(51.0/255.0);
color magenta; 
magenta := red + blue; 
color yellow;
yellow := (236.0/255.0, 228.0/255.0, 18.0/255.0);


def dodrawpatharr text txt  = 
    begingroup; 
       
       pair _arr_aa, _arr_ad; 
       path _pth_fin, _pth_draw; 
       
       _arr_aa := point(length(_apth)) of _apth;
       _apth_alen := arclength(_apth);
       _pth_fin := (subpath(arctime(_apth_alen - ahlength) of _apth, 
                           length(_apth)) of _apth) 
		  shifted -_arr_aa; 
       _pth_draw := subpath(0, arctime(_apth_alen - arrowasp*ahlength) of _apth)  of _apth; 
       _arr_ad := point(length(_pth_draw)) of _pth_draw; 
       draw _pth_draw txt; 
       path _pth_a, _pth_b; 
       _pth_a := (_pth_fin rotated (0-ahangle/2)) shifted _arr_aa; 
       _pth_b := reverse((_pth_fin rotated (0+ahangle/2)) shifted _arr_aa); 
       
       path _arr_contour; 
       _dir_end := angle(_arr_aa - _arr_ad) + 90; 
       _arr_contour :=  _arr_ad {dir _dir_end}
                    .. (point(0)of _pth_a) 
		    -- _pth_a -- _pth_b 
		    -- (point (length(_pth_b)) of _pth_b)
		    .. {dir _dir_end}_arr_ad; 
       %_arr_contour := _pth_b 
       %  ..  {dir _dir_end} _arr_ad {dir _dir_end}   
       %	 .. _pth_a; 
       fill buildcycle(_arr_contour) txt; % withpen scaled 1pt; 
       if arrdebug > 0: 
          pickup pencircle scaled 0.5pt; 
          draw _arr_contour withcolor red;
       fi; 
       endgroup; 
enddef;

def dodrawpathdblarr text txt  = 
    begingroup; 
       
       pair _arr_aa, _arr_ad, _arr_ba, _arr_bd; 
       path _pth_fin, _pth_draw, _pth_beg; 
       
       _arr_aa := point(length(_apth)) of _apth;
       _arr_ba := point(0) of _apth;
       _apth_alen := arclength(_apth);
       _pth_fin := (subpath(arctime(_apth_alen - ahlength) of _apth, 
                           length(_apth)) of _apth) 
		  shifted -_arr_aa; 
       _pth_beg := (subpath(0, arctime(ahlength) of _apth) of _apth) 
		  shifted -_arr_ba; 
       _pth_draw := subpath(arctime(arrowasp*ahlength) of _apth, 
            arctime(_apth_alen - arrowasp*ahlength) of _apth)  of _apth; 
       _arr_ad := point(length(_pth_draw)) of _pth_draw; 
       _arr_bd := point(0) of _pth_draw; 
       draw _pth_draw txt; 
       path _pth_ea, _pth_eb, _pth_ba, _pth_bb; 
       _pth_ea := (_pth_fin rotated (0-ahangle/2)) shifted _arr_aa; 
       _pth_eb := reverse((_pth_fin rotated (0+ahangle/2)) shifted _arr_aa); 
       _pth_ba := reverse(_pth_beg rotated (0-ahangle/2)) shifted _arr_ba; 
       _pth_bb := ((_pth_beg rotated (0+ahangle/2)) shifted _arr_ba); 
       
       path _arr_contour_e, _arr_contour_b; 
       _dir_end := angle(_arr_aa - _arr_ad) + 90; 
       _dir_beg := angle(_arr_bd - _arr_ba) - 90; 
       _arr_contour_e :=  _arr_ad {dir _dir_end}
                    .. (point(0)of _pth_ea) 
		    -- _pth_ea -- _pth_eb 
		    -- (point (length(_pth_eb)) of _pth_eb)
		    .. {dir _dir_end}_arr_ad; 
       _arr_contour_b :=  _arr_bd {dir _dir_beg}
                    .. (point(0)of _pth_ba) 
		    -- _pth_ba -- _pth_bb 
		    -- (point (length(_pth_bb)) of _pth_bb)
		    .. {dir _dir_beg}_arr_bd; 
%       _arr_contour_e := _pth_ea -- _pth_eb 
%         ..  {dir _dir_end} _arr_ad {dir _dir_end}   
%	 .. (point(0) of _pth_ea); 
%       _arr_contour_b := _pth_ba -- _pth_bb 
%         ..  {dir _dir_beg} _arr_bd {dir _dir_beg}   
%	 .. (point(0) of _pth_ba); 
       fill buildcycle(_arr_contour_e) txt; % withpen scaled 1pt; 
       fill buildcycle(_arr_contour_b) txt; % withpen scaled 1pt; 
       if arrdebug > 0: 
          pickup pencircle scaled 0.5pt; 
          draw _arr_contour_b withcolor red; 
          draw _arr_contour_e withcolor red; 
       fi; 
       endgroup; 
enddef; 

def drawpatharr expr p = _apth:= p; dodrawpatharr enddef; 

def drawpathdblarr expr p = _apth:= p; dodrawpathdblarr enddef; 

def dosinpath (expr n, np) = 
    %(0, 0) -- (0.5, 1.0) -- (1.0, 0.0)
    ((0, 0)
    for i = 1 upto np: 
       .. (i/np, sind(i/np*n*180))
    endfor) 
enddef; 

def sinpath(expr a, b, h, n, np) = 
     ((dosinpath(n, np)  xscaled (abs(b - a))) yscaled h) rotated (angle(b - a)) shifted a 
enddef; 


def docoilpath (expr n, np, v) = 
    %(0, 0) -- (0.5, 1.0) -- (1.0, 0.0)
    ((0, 0)
    for i = 1 upto np: 
       .. (10*i/np*(1.0)*(1 + v*(cosd(n*360) - 1.0)) - 10*v*(cosd(i/np*n*360) - 1.0), sind(i/np*n*360))
    endfor) 
enddef; 


def coilpath(expr a, b, h, v, n, np) = 
     ((docoilpath(n, np, v)  xscaled (abs(b - a)/10)) yscaled h) rotated (angle(b - a)) shifted a 
enddef; 
 
def arcpoint(expr p, i, np) = 
    (point (arctime((arclength(p)/np)*i) of p) of p)
enddef; 

def dirvector(expr p, i, np) = 
    unitvector(arcpoint(p, i, np) - arcpoint(p, i - 1, np))
enddef; 

def sinifiedpath(expr p, nw, h, np) = 
    (point 0 of p)
    for _i=1 upto np: 
        .. (arcpoint(p, _i, np) + h*sind((_i/np)*180.0*nw)*(dirvector(p, _i, np) rotated 90))
    endfor
enddef; 

def coiledpath(expr p, h, v, nw, np) = 
    (point 0 of p)
    for _i=1 upto np: 
        .. (arcpoint(p, _i, np) + h*sind((_i/np)*180.0*nw)*(dirvector(p, _i, np) rotated 90)
	           + h*v*dirvector(p, _i, np)*((cosd((_i/np)*nw*180.0)-1) -_i/np*(cosd(nw*180.0) - 1) ) )
    endfor
enddef; 

def drawairypath(expr p, np, gap, fgcol, bgcol) = 
    begingroup; 
       path _pth_left, _pth_before, _pth_after, _pth_before; 
       _arc_int[0] := 0.0; 
       _pth_left := p; 
       _len_left := arclength(_pth_left);
       _lstep := arclength(p) / 100; 
       _n_int := 1; 
       _arc_cur := 0.0;
       forever: exitunless (_len_left > _lstep);
           _t_c := arctime(_arc_cur) of _pth_left; 
	   _t_n := arctime(_arc_cur + _lstep) of _pth_left;
	   _pth_before := subpath(_t_c, _t_n) of _pth_left; 
	   _pth_after  := subpath(_t_n, length(_pth_left)) of _pth_left;  
	   pair _it_times; 
	   _it_times := _pth_before intersectiontimes _pth_after; 
	   message "len:" & decimal length(_pth_before); 
	   message "left: " & decimal _len_left; 
	   message "intersection: " & decimal xpart(_it_times) & " " & decimal ypart(_it_times); 
	   if (ypart(_it_times) > 0.001) and (xpart(_it_times) > 0.001): 
	      %_gap := 0.0; 
	      path _pth_a; 
	      _pth_a := subpath(0, xpart(_it_times)) of _pth_before;
	      _len_a :=  arclength(_pth_a);
	      _arc_int[_n_int] := _arc_cur + _len_a; 
	      _arc_cur := _arc_int[_n_int]; 
	      _n_int := _n_int + 1;
	      _len_left := _len_left - _len_a; 
	      message "segment: " & decimal _arc_int[_n_int - 1];
	   else:
	      _arc_cur := _arc_cur + _lstep; 
	      _len_left := _len_left - _lstep;
	     message "increment: " & decimal _arc_cur & " " & decimal _len_left; 
	   fi; 
       endfor; 
       _arc_int[_n_int] := arclength(_pth_left); 
       message "split points found";
       _a_s[0] := 0; 
       _e_s[0] := 0; 
       for _i_int = 1 upto _n_int - 1: 
           _a_s[_i_int] := _arc_int[_i_int] + gap/2.0;
	   message "start: " & decimal _a_s[_i_int]; 
       endfor; 
       for _i_int = 0 upto _n_int - 1: 
	   _e_s[_i_int] := _arc_int[_i_int + 1] - gap/2.0;
	   message "end: " & decimal _e_s[_i_int]; 
       endfor; 
       _e_s[_n_int - 1] := _arc_int[_n_int]; 
       for _i_int = 0 upto _n_int - 1:
           _t_c := arctime(_a_s[_i_int]) of _pth_left; 
	   _t_n := arctime(_e_s[_i_int]) of _pth_left; 
	   message "draw segment" & decimal _t_c & " " & decimal _t_n ;  
	   draw subpath(_t_c, _t_n) of _pth_left withcolor fgcol; 
       endfor; 
	   %message "len:" & decimal length(_pth_before); 
	   %message "intersection: " & decimal xpart(_it_times) & " " & decimal ypart(_it_times); 
	   %path _pth_draw, _pth_a, _pth_b; 
	   %_pth_a := subpath(0, xpart(_it_times)) of _pth_left; 
	   %_pth_b := subpath(xpart(_it_times), length(_pth_left)) 
	   %          of _pth_left; 
	   %_pth_draw := subpath(0, 
	   %                arctime(arclength(_pth_a) -_gap/2) of _pth_a) 
           %	            of _pth_a;
	   %_pth_left := subpath(arctime(_gap/2) of _pth_b, 
	   %                 length(_pth_b)) of _pth_b;
	   %draw _pth_draw withcolor fgcol; 
           %_len_left := arclength(_pth_left); 
       %endfor; 
       %draw _pth_left withcolor fgcol; 
    endgroup; 
enddef; 

def setax(suffix ax)(expr o, theta, phi, unt) = 
    begingroup;
       ax[1] := -sind(phi) *unt; 
       ax[2] := cosd(phi)  *unt; 
       ax[3] := 0;
       ax[4] := -sind(theta)*cosd(phi) * unt;
       ax[5] := -sind(theta)*sind(phi) * unt;
       ax[6] := cosd(theta) * unt; 
       ax[7] := xpart(o); 
       ax[8] := ypart(o); 
       ax[9]  := cosd(phi) * cosd(theta);  
       ax[10] := sind(phi) * cosd(theta); 
       ax[11] := sind(theta); 
    endgroup; 
enddef; 


def axpr(suffix axn)(expr x, y, z) = 
    (x * axn[1] + y*axn[2] + z*axn[3] + axn[7], x*axn[4] + y*axn[5] + z*axn[6] + axn[8])
enddef; 

def dist(suffix axn)(expr x, y, z) = 
    axn[9] * x + axn[10]*y + axn[11] * z
enddef; 

def drawaxes(expr o, xp, yp, xm, ym) = 
    begingroup; 
       drawpatharr (o shifted (xm, 0)) -- (o shifted (xp, 0)); 
       drawpatharr (o shifted (0, ym)) -- (o shifted (0, yp)); 
    endgroup; 
enddef; 

def mkpath(suffix pth, pt_set)(expr N_set) = 
    pth := pt_set[0] 
    for _iii = 1 upto N_set: 
        .. pt_set[_iii]
    endfor; 
enddef; 

def roundframe(expr wid, hei, rcorn) = 
    %buildcycle((0, 0) -- (wid, 0) -- (wid, hei) -- (0, hei) -- (0, 0))
    buildcycle (
       (  ((rcorn, 0) -- (wid - rcorn, 0))
       -- ((subpath (0.75, 1.0)*length(fullcircle) of fullcircle) 
               scaled 2rcorn shifted (wid - rcorn, rcorn))
       -- ((wid, rcorn) -- (wid, hei - rcorn))
       -- ((subpath (0.0, 0.25)*length(fullcircle) of fullcircle) 
               scaled 2rcorn shifted (wid - rcorn, hei - rcorn))
       -- ((wid - rcorn, hei) -- (rcorn, hei))
       -- ((subpath (0.25, 0.5)*length(fullcircle) of fullcircle) 
               scaled 2rcorn shifted (rcorn, hei - rcorn))
       -- ((0, hei - rcorn) -- (0, rcorn))
       -- ((subpath (0.5, 0.75)*length(fullcircle) of fullcircle) 
               scaled 2rcorn shifted (rcorn, rcorn))
       -- (rcorn, 0)
       )
    ) 
enddef; 

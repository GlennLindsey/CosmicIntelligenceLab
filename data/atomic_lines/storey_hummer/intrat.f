      program intrat
c
*******************************************************************************
c
c          Reads emissivities for all transition with n.le.ncut from
c          the input files supplied with the program and computes 
c          INTensity RATios of HYDROGENIC recombination lines for
c          specified transitions at temperatures and densities in input
c          files. The parameter 'ncut' is specified in the tables and 
c          is currently set to 25. Extensive facilities are provided 
c          for interactive two-dimensional interpolation to specified 
c          temperatures and densities.  Interpolations are made to 
c          four orders (currently 2,3,4 5) to give some impression 
c          of the reliability of the interpolation, and can be made in
c          either the intensity ratios (or emissivities), or in the 
c          logarithms of these quantities.  The interpolates are output 
c          together following the symbol "r(i)=" or "r(l)=", which 
c          indicates linear or logarithmic interpolation. Experience to 
c          date has shown that the logarithmic interpolations are 
c          almost always more reliable, and is essential for emissivities. 
c        
c          Total recombination coefficients are also available and
c          can be accessed by responding to the cue concerning "alpha-tot"
c          or by specifying "a" from the menu after the table display. 
c          Interpolation in this table is identical to that for other
c          tables; for low temperatures and high densities logarithmic
c          interpolation is necessary.
c
c          Input files are named e1a.d, e1b.d, e2a.d, e2b.d,...,
c          e8a.d, e8b.d, and contain emissivities for cases A nd B
c          of all hydrogenic ions through oxygen. (at present, only
c          'b'-files, but the remainder are being computed).
c
c          Output appears also in file intrat_zion_case.d.
c
c
c          *********************************************************
c          *                                                       *
c          *     !!!!!  PRELIMINARY VERSION 1.0  !!!!!             *
c          *                                                       *      
c          *                 19 february 1993 (0.0)                *
c          *                                                       *
c          *                   15 july 1993
c          *                                                       *
c          *     please send error reports and suggestions for     *
c          *          improvements to D.G.Hummer at                *
c          *                                                       *
c          *               dgh@usm.uni-muenchen.de                 *
c          *                                                       *
c          *                 (49)-(0)89-9220-9441                  *
c          *                                                       *
c          *********************************************************
c
c
c          respond to prompts for the following input:
c               ZION  -  charge of recombining ion (<9)
c               CASE  -  A or B
c               DMIN  -  minimum vlue of electron density; data for six higher
c                        values will appear in output tables
c               N_upper, N_lower, N_upper_ref, N_lower_ref - upper and
c                        lower principal quantum numbers for transition
c                        of interest and of reference transition. For
c                        the emissivity of the transition, set
c                            N_upper_ref = N_lower_ref = 0 
c                        note:  N_upper = -1 allows minimum density to be reset
c                        and setting all four N's = 0 ends run
c
c
c          after desired portion of table is displayed, respond to prompts for
c               - 2-d interpolation of intensity ratio or emissivity (i)
c               - 2-d logarithmic interpolation of table (l)
c               - new transition(s) (n)
c               - alpha-tot (a)
c               - set new value of minimum density displayed in table (d)
c               - end run (e)
c     
c          P.J.Storey and D.G.Hummer, January 1993 ; see MNRAS,xxxxx,1993
c
*******************************************************************************

      implicit real*4(a-h,o-z)
      character*1 zion,case,ques
      character*2 name
      dimension dens(15),temp(15),e(300,15,15),r(15,15),x(15),y(15),
     &          ni(5),cx(5),cy(5),ri(5),f(2,15),a(15,15)
      data max/4/ni/2,3,4,5,6/       ! interpolation parameters
c
c          identify ion and case
c
1     write(*,5)
5     format(//'           ****** WELCOME TO INTRAT ******'//
     &'  please type ZION and CASE, separated by one space'//)
      read(*,10) zion,case
10    format(a1,1x,a1)
      if(case.eq.'a'.or.case.eq.'A') then
           ncase=0
      else
           ncase=1
      endif
c
c          open i/o files
c
      name=zion//case
      open(unit=15,file='e'//name//'.d',status='old')
      open(unit=16,file='intrat'//name//'.d',status='new')
c
c          output ion, case and range specifications
c
      write(*,20) zion,case
      write(16,20) zion,case
20    format('  zion= ',a1,'  case= ',a1/)
      write(*,21) 'e'//name//'.d'
21    format('  please wait - data file ',a5,' is being loaded')
c
c          read secondary file for this ion and case
c
      read(15,*) ntemp,ndens
      do 101 ia=1,ntemp
           do 100 ib=1,ndens
                read(15,25) dens(ib),temp(ia),ncut
25              format(1x,e10.3,5x,e10.3,13x,i2)
                ne=ncut*(ncut-1)/2
                read(15,30) (e(j,ia,ib),j=1,ne)
30              format((8e10.3))
100        continue  
101   continue
      read(15,*) ((a(i,j),i=1,ndens),j=1,ntemp)
      write(*,31)                        
31    format('  data input complete'//)
      write(*,32) (temp(i),i=1,ntemp)
32    format(' temperatures:'/(1p7e10.3))
      write(*,33) (dens(i),i=1,ndens)
33    format(/' densities:'/(1p7e10.3))
      write(*,34) (ni(i),i=1,max)
      write(16,34) (ni(i),i=1,max)
34    format(/' interpolation orders=',5i3)

c
c          interpolation variables
c
      do 102 i=1,ndens
           x(i)=alog10(dens(i))
102   continue
      do 103  i=1,ntemp
           y(i)=sqrt(temp(i))
           f(1,i)=1.0          ! f is emissivity smoothing function in temp 
           f(2,i)=y(i)
103   continue
c
c          limit output to desired minimum density plus six higher values
c
38    write(*,39)
39    format(/'  please type minimum density to be considered'/)
      read(*,*,err=38) dmin
      m0=1
      do 104 i=1,ndens
           if(dens(i).le.dmin) then
                m0=i
           endif
104   continue
      m1=m0+6
      if(m1.gt.ndens) then
           m1=ndens
      endif
c
c          direct access to alpha-tot
c
      write(*,40) 
40    format(/' start with line-ratios/emiss. (l) or alphas (a)?')
      read(*,41) ques
41    format(a1)
c
c     load alpha-tot table in table array
c
42    if(ques.eq.'a'.or.ques.eq.'A') then
           do 106 it=1,ntemp
                do 105 id=1,ndens
                     r(it,id)=a(id,it)
105             continue 
106        continue 
           ns=1
           goto 60
      endif
c
c          choose transition of interest (and standard transition if wanted)
c
45    write(*,50)
50    format(/' n_upper, n_lower, n_upper_ref, n_lower_ref'/)
      read(*,*,err=45) nu,nl,nus,nls
      if(nu.lt.0) then                   ! change minimum density
           go to 38
      endif
      if(nu.eq.0.or.nl.eq.0) then        ! end program
           stop
      endif
      if(nu.le.nl.or.nu.gt.ncut) then    ! check order of levels
           goto 45
      endif
      if(nus.lt.nls.or.nus.gt.ncut) then ! check order of levels
           goto 45
      endif
      if((nu.gt.2.and.nl.eq.1.and.ncase.eq.1).or.(nus.gt.2.and.nls.eq.1.
     &and.ncase.eq.1)) then  ! Ly beta,.. 
           go to 45
      endif
      write(16,52) nu,nl,nus,nls
52    format(//' nu=',i3,' nl=',i3,' nus=',i3,' nls=',i3/)
c
c          set keys to locate transitions of interest
c
      if((nus+nls).eq.0) then
           ns=2
           ks=999
      else
           ns=1
           ks=(((ncut-nus)*(ncut+nus-1))/2)+nls
      endif
      k=(((ncut-nu)*(ncut+nu-1))/2)+nl
c
c          calculate desired intensity ratio (or emissivity if nus=nls=0)
c
      do 108 it=1,ntemp
           do 107 id=1,ndens
                if(ns.eq.1) then
                     r(it,id)=e(k,it,id)/e(ks,it,id)
                else
                     r(it,id)=e(k,it,id)
                endif
107        continue
108   continue
c
c          output table of line intensity (ratios) to screen and file
c
60    write(*,70) (dens(i),i=m0,m1)
      write(16,70) (dens(i),i=m0,m1)
70    format(' dens:     ',1p7e9.2)
      write(*,71)
      write(16,71)
71    format(' temp      ')
      do 109 i=1,ntemp
           write(*,72) temp(i),(r(i,j),j=m0,m1)
           write(16,72) temp(i),(r(i,j),j=m0,m1)
72         format(1pe9.2,2x,7e9.2)
109   continue
c
c          interpolate in r-table
c
78    write(*,80)
80    format(/' interpolate(i), interpolate log(l), new n(n), 
     &alpha-tot(a), new min density(d)'/' exit(e)?')
      read(*,82,err=78) ques
82    format(a1)
      if(ques.eq.'e'.or.ques.eq.'E') then
           stop
      endif
      if(ques.eq.'a'.or.ques.eq.'A') then
           goto 42
      endif
      if(ques.eq.'d'.or.ques.eq.'D') then
           goto 38
      endif
      if(ques.eq.'n'.or.ques.eq.'N') then
           goto 45
      endif
      if(ques.eq.'i'.or.ques.eq.'I'.or.ques.eq.'l'.or.ques.eq.'L') then
           if(ques.eq.'i'.or.ques.eq.'I') then
                nt=0
           else
                nt=1
           endif
83         write(*,84)
84         format(' please type desired temperature and density')
           read(*,*,err=83) xt,xd
           if(xt.lt.temp(1).or.xt.gt.temp(ntemp).or.xd.lt.dens(1).
     &     or.xd.gt.dens(ndens)) then
                write(*,85)
85              format(' requested temp/dens not in table')
                goto 78
           endif
           xp=alog10(xd)           ! interpolate in log(dens)
           yp=sqrt(xt)             ! interpolate in temp**0.5
c
c          find interpolation box 
c
           i=1
86         if(xp.ge.x(i).and.xp.le.x(i+1)) then
                goto 88
           else
                i=i+1
                if(i.eq.ndens) then
                     stop 'dens overflow'
                endif
                goto 86
           endif
88         i0=i
           j=1
90         if(yp.ge.y(j).and.yp.le.y(j+1)) then
                goto 92
           else
                j=j+1
                if(j.eq.ntemp) then
                     stop 'temp overflow'
                endif
                goto 90
           endif
92         j0=j
c
c          interpolate to orders 2,3,4,5 in both directions
c
           do 116 int=1,max
                nint=ni(int)             ! interpolation order
                nint1=nint-1
                nof=nint1/2
c
c          shift i0 to nearest box boundary in each direction if nint is odd
c
                if(nint.eq.3.or.nint.eq.5.or.nint.eq.7) then  ! note ODD order
                     if((xp-x(i0)).gt.(x(i0+1)-xp)) then
                           is=i0+1-nof
                     else
                           is=i0-nof
                     endif
                     if((yp-y(j0)).gt.(y(j0+1)-yp)) then
                          js=j0+1-nof
                     else
                          js=j0-nof
                     endif
                else
                     is=i0-nof
                     js=j0-nof
                endif
c
c          ensure that interpolation box lies in table
c
                if(is.lt.1) then
                     is=1
                endif
                if((is+nint1).gt.ndens) then
                     is=ndens-nint1
                endif
                if(js.lt.1) then
                     js=1
                endif
                if((js+nint1).gt.ntemp) then
                     js=ntemp-nint1
                endif
c
c          nint**2-point interpolation
c
                do 111 k=1,nint
                     i=is+k-1
                     cx(k)=1.0
                          do 110 kp=1,nint
                               if(kp.ne.k) then
                                    ip=is+kp-1
                                    cx(k)=cx(k)*(xp-x(ip))/(x(i)-x(ip))
                               endif
110                       continue
111             continue
                do 113 k=1,nint
                     j=js+k-1
                     cy(k)=1.0
                          do 112 kp=1,nint
                               if(kp.ne.k) then
                                    jp=js+kp-1
                                    cy(k)=cy(k)*(yp-y(jp))/(y(j)-y(jp))
                               endif
112                       continue
113             continue
                rint=0.0
                do 115 kx=1,nint
                     do 114 ky=1,nint
                          if((js+ky-1).gt.ntemp.or.(is+kx-1).gt.ndens)
     .                         then
                               stop 'final loop error'
                          endif
                          rrr=r(js+ky-1,is+kx-1)*f(ns,js+ky-1) ! smoothing ftn  
                          if(nt.ne.0) then
                               rrr=log(rrr)
                          endif
                          rint=rint+cx(kx)*cy(ky)*rrr
114                  continue
115             continue
                ri(int)=rint
                if(nt.ne.0) then
                     ri(int)=exp(ri(int))
                endif
                if(ns.eq.2) then
                     ri(int)=ri(int)/yp ! remove smoothing function = temp**.5
                endif
116        continue                              ! end nint-loop
           write(*,94) xt,xd,ques,(ri(i),i=1,max)
           write(16,94) xt,xd,ques,(ri(i),i=1,max)
94         format(/' Te=',1pe10.3,' Ne=',e10.3,'  r(',a1,')=',4e9.2)
           goto 78
      endif
c
c          meaningless character - try again
c
      goto 78
c
      end

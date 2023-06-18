% Iterative finite difference program for 2D viscous deformation, GNU Octave version
% This code is free software under the Creative Commons CC-BY-NC-ND license
% Published in W.R.Halter, E.Macherel, and S.M.Schmalholz (2022) JSG, https://doi.org/10.1016/j.jsg.2022.104617
clear all, close all, clc
% Additional functions perfoming interpolations on the numerical grid
function A1 = n2c(A0) % Interpolation of nodal points to center points
A1      = (A0(2:end,:) + A0(1:end-1,:))/2;
A1      = (A1(:,2:end) + A1(:,1:end-1))/2;
end
function A2 = c2n(A0) % Interpolation of center points to nodal points
A1    	= zeros(size(A0,1)+1,size(A0,2));
A1(:,:)	= [1.5*A0(1,:)-0.5*A0(2,:); (A0(2:end,:)+A0(1:end-1,:))/2; 1.5*A0(end,:)-0.5*A0(end-1,:)];
A2   	= zeros(size(A1,1),size(A1,2)+1);
A2(:,:)	= [1.5*A1(:,1)-0.5*A1(:,2), (A1(:,2:end)+A1(:,1:end-1))/2, 1.5*A1(:,end)-0.5*A1(:,end-1)];
end
% Beginning of the actual Script
ps          = 1;    % ps = 1 models pure shear; ps = 0 models simple shear
etaRatio    = 1000;   % Viscosity ratio (matrix/inclusion) %CT
a           = 0.2; b = a/2; phi = 30/180*pi; % Parameters defining elliptical inclusion
% Numerical parameters
nx          = 151;              ny          = nx;                   % Numerical resolution
Lx          = 1;                Ly          = Lx;                   % Model dimension
dx          = Lx/(nx-1);        dy          = Ly/(ny-1);            % Grid spacing
x           = [-Lx/2:dx:Lx/2];  y           = [-Ly/2:dy:Ly/2];      % Coordinate vectors
x_vx        = [x(1)-dx/2,( x(1:end-1) + x(2:end) )/2,x(end)+dx/2];  % Horizontal vector for Vx which is one more than basic grid
y_vy        = [y(1)-dy/2,( y(1:end-1) + y(2:end) )/2,y(end)+dy/2];  % Vertical   vector for Vy which is one more than basic grid
% 3 numerical grids due to staggered grid
[X,Y]       = ndgrid(x,y);      [X_vx,Y_vx] = ndgrid(x_vx,y); [X_vy,Y_vy] = ndgrid(x,y_vy);
% Physical parameters
eta_B       = 1;                n_exp       = 1;
s_ref       = 1.5;              D_B         = 1;     
% Initialization
P           = zeros(nx,  ny);   ETA         = eta_B*ones(nx, ny);
% Define elliptical inclusion
inside      = ((X*cos(phi)+Y*sin(phi)).^2/a^2 + (X*sin(phi)-Y*cos(phi)).^2/b^2 < 1);
ETA(inside) = eta_B/etaRatio;   etamin      = 0.1*min(ETA(:));
for smo=1:2; Ii  = [2:nx-1]; Ij  = [2:ny-1]; % Smoothing of the initial viscosity field
    ETA(Ii,:)    = ETA(Ii,:) + 0.4*(ETA(Ii+1,:)-2*ETA(Ii,:)+ETA(Ii-1,:));
    ETA(:,Ij)    = ETA(:,Ij) + 0.4*(ETA(:,Ij+1)-2*ETA(:,Ij)+ETA(:,Ij-1));
end
ETA_L       = ETA;              ETA_PL      = ETA;
% Boundary condition
if     ps==1,   VX    	= -D_B*X_vx;    	VY  	= D_B*Y_vy;
elseif ps==0,   VX   	=  D_B*Y_vx;    	VY   	=   0*Y_vy; end
% Parameters for pseudo-transient iterations
tol         = 5e-6;             error       = 10*tol;
dpt_P       = 50  *min(dx,dy)^2/max(max(ETA)); % Pseudo time step pressure
dpt_V       = 0.05*min(dx,dy)^2/max(max(ETA)); % Pseudo time step velocity
Pold        = P;                iter        = 0;
while error>tol; iter = iter+1; % START of iteration loop
    DXX                 = diff(VX,1,1)/dx;                          % Eq (1)
    DYY                 = diff(VY,1,2)/dy;                          % Eq (2)
    DXY                 = 1/2*( diff(VX(2:end-1,:),1,2)/dy ...
                              + diff(VY(:,2:end-1),1,1)/dx );       % Eq (3)
    TXX                 = 2.*ETA.*DXX;                              % Eq (4)
    TYY                 = 2.*ETA.*DYY;                              % Eq (5)
    TXY                 = 2.*n2c(ETA).*DXY;                         % Eq (6)
    SXX                 = -P(:,2:end-1)+TXX(:,2:end-1);             % Eq (7)
    SYY                 = -P(2:end-1,:)+TYY(2:end-1,:);             % Eq (8)
    RES_P               = -( diff(VX,1,1)/dx + diff(VY,1,2)/dy );   % Eq (16)
    RES_VX              = diff(SXX,1,1)/dx + diff(TXY,1,2)/dy;      % Eq (17)
    RES_VY              = diff(SYY,1,2)/dy + diff(TXY,1,1)/dx;      % Eq (18)
    P                   = P                   + dpt_P*RES_P;        % Eq (22)
    VX(2:end-1,2:end-1) = VX(2:end-1,2:end-1) + dpt_V*RES_VX;       % Eq (23)
    VY(2:end-1,2:end-1) = VY(2:end-1,2:end-1) + dpt_V*RES_VY;       % Eq (24)
    TII                 = sqrt(0.5*(TXX.^2 + TYY.^2 + 2*c2n(TXY).^2)); % Eq (13)
    if n_exp>1 % Power-law viscosity
        ETA_PL_it       = ETA_PL;           % Viscosity of previous iteration step
        ETA_PL          = ETA_L.*(TII/s_ref).^(1-n_exp);            % Eq (12)
        ETA_PL          = exp(log(ETA_PL)*0.5+log(ETA_PL_it)*0.5);
        ETA             = 1./( 1./ETA_L + 1./ETA_PL );              % Eq (14)
        ETA(inside)     = ETA_L(inside);    % Power-law viscosity only applied to matrix
        ETA(ETA<etamin) = etamin;           % Minimum viscosity
    end
    error = max([max(abs(dpt_V*RES_VX(:))), max(abs(dpt_V*RES_VY(:))), max(abs(dpt_P*RES_P(:)))]);
    disp(error);
    if mod(iter,2000)==0 % Visualization during calculation

        subplot(321),pcolor(X,Y,P),shading interp,axis equal,axis tight,colorbar,colormap(jet)
        title({'A) Pressure [ ], P '; ['Iteration: ',num2str(iter),'; error: ',num2str(error)]}),ylabel('Height [ ]')

        subplot(322),pcolor(X,Y,log10(ETA)), shading interp, colormap(jet), hold on
        title({'B) Viscosity [ ], log_{10}(\eta),';'and velocity arrows'})
        st = 15; VXC = (VX(1:end-1,:)+VX(2:end,:))/2; VYC = (VY(:,1:end-1)+VY(:,2:end))/2;
        quiver(X(1:st:end,1:st:end),Y(1:st:end,1:st:end),VXC(1:st:end,1:st:end),VYC(1:st:end,1:st:end),'w');
        axis equal,axis tight,colorbar,axis([-Lx/2,Lx/2,-Ly/2,Ly/2]),hold off

        subplot(323),pcolor(X,Y,TII),shading interp,axis equal,axis tight,colorbar,colormap(jet)
        title(['C) 2nd stress invariant [ ], Tii']), xlabel('Width [ ]'), ylabel('Height [ ]')

        subplot(324),pcolor(X,Y,DXX),shading interp,axis equal,axis tight,colorbar,colormap(jet) %CT
        title(['D) Horizontal strain rate [ ], Dxx']), xlabel('Width [ ]')

        subplot(325),pcolor(X_vx,Y_vx,VX),shading interp,axis equal,axis tight,colorbar,colormap(jet)
        title(['E) Horizontal velocity [ ], Vx']), xlabel('Width [ ]'), ylabel('Height [ ]')

        subplot(326),pcolor(X_vy,Y_vy,VY),shading interp,axis equal,axis tight,colorbar,colormap(jet)
        title(['F) Vertical velocity [ ], Vy']), xlabel('Width [ ]'), ylabel('Height [ ]')

        set(gcf,'position',[281 47.4000 975.2000 724]), drawnow
        print ('title_axis.png','-S1000,1000'); %CT
    end
end % END of iteration loop

function [yFun, uFun, out] = kkt_ex1_L_simple(ndof, varargin)
%KKT_EX1_L_SIMPLE  Knappes FEM-Solver-Interface (y,u) fuer ex1-KKT auf L-Domaene.
%
%   [yFun, uFun, out] = kkt_ex1_L_simple(ndof, ...)
%
%   Loest das (manufactured) Optimalitaetssystem aus ex1.py auf der L-Domaene
%       Omega = [0,1]^2 \ ([0.5,1] x [0.5,1])
%   mit P1-FEM (PDE Toolbox Assemblierung) und linearer Algebra auf Blockebene.
%   Omega = [0,1]^2 \ ([0.5,1] x [0.5,1])
%
% Manufactured Problem (angelehnt an ex1.py):
%   -Laplace(y) + (1/lambda) * p = 0
%   -Laplace(p) - y              = -y_d
%   u = -(1/lambda) * p
% mit y_d = (1 + 4*lambda*pi^4) * sin(pi x) * sin(pi y)

p = inputParser;
p.addRequired('ndof', @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 10);
p.addParameter('Lambda', 0.01, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
p.addParameter('MaxMeshIter', 10, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('MeshTolRel', 0.10, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('SaveMat', '', @(s) isstring(s) || ischar(s));
p.parse(ndof, varargin{:});

lambda = double(p.Results.Lambda);

% ------------------------------------------------------------
% 1) Geometrie + Mesh (L-Domaene)
% ------------------------------------------------------------
model = createpde(1);
geometryFromEdges(model, l_domain_decsg());

[mesh, Hmax, meshInfo] = calibrate_mesh(model, round(ndof), p.Results.MeshTolRel, p.Results.MaxMeshIter);

nodes = mesh.Nodes;       % 2 x N Knoten der Dreieicke
elements = mesh.Elements; % 3 x Nt Ein Dreieick wird durch 3 Eckknoten beschrieben
nNodes = size(nodes, 2);
nodeX = nodes(1, :)';
nodeY = nodes(2, :)';

% ------------------------------------------------------------
% 2) FEM-Matrizen: Steifigkeit K, Masse M, Last Fyd
% ------------------------------------------------------------
% Laplace-Operator: K
specifyCoefficients(model, 'm', 0, 'd', 0, 'c', 1, 'a', 0, 'f', 0);
FEM0 = assembleFEMatrices(model);
K = FEM0.K; % Steifigkeitsmatrix K

% echte L2-Massenmatrix M (robust ueber MATLAB-Versionen)
M = assemble_mass_matrix_p1(nodes, elements);

% Lastvektor fuer y_d
specifyCoefficients(model, 'm', 0, 'd', 0, 'c', 1, 'a', 0, 'f', @(loc,~) y_desired(loc.x, loc.y, lambda));
FEMyd = assembleFEMatrices(model);
Fyd = FEMyd.F;

% ------------------------------------------------------------
% 3) KKT-Blocksystem fuer (y,p)
% ------------------------------------------------------------
A = [K, (1/lambda) * M;
     -M,          K];

b = [zeros(nNodes, 1);
     -Fyd];

% ------------------------------------------------------------
% 4) Dirichlet-Randwerte (manufactured) + Elimination
% ------------------------------------------------------------
bnd = boundary_nodes_from_elements(elements);
int = setdiff((1:nNodes)', bnd);

yB = y_exact(nodeX(bnd), nodeY(bnd)); %Randwerte der Knoten
pB = p_exact(nodeX(bnd), nodeY(bnd), lambda);

idxY_b = bnd;
idxP_b = nNodes + bnd;
idxY_i = int;
idxP_i = nNodes + int;

idxB = [idxY_b; idxP_b];
idxI = [idxY_i; idxP_i];

xKnown = zeros(2*nNodes, 1); %exakte Werte an den Kanten setzten
xKnown(idxY_b) = yB;
xKnown(idxP_b) = pB;

x = zeros(2*nNodes, 1);
if isempty(idxI)
    % Extrem grobes Mesh: nur Randknoten
    x(idxB) = xKnown(idxB);
else
    rhs = b(idxI) - A(idxI, idxB) * xKnown(idxB);
    x(idxI) = A(idxI, idxI) \ rhs;
    x(idxB) = xKnown(idxB);
end

% ------------------------------------------------------------
% 5) u aus p, Funktions-Handles, Metriken
% ------------------------------------------------------------
yNodal = x(1:nNodes);
pNodal = x(nNodes+1:end);
uNodal = -(1/lambda) * pNodal;

tri = triangulation(elements', nodes');

yFun = @(xq, yq) fem_eval(tri, yNodal, xq, yq);
uFun = @(xq, yq) fem_eval(tri, uNodal, xq, yq);

metrics = compute_metrics(yFun, uFun, lambda);

out = struct();
out.lambda = lambda;
out.model = model;
out.mesh = mesh;
out.hmax = Hmax;
out.meshInfo = meshInfo;
out.K = K;
out.M = M;
out.Fyd = Fyd;
out.A = A;
out.b = b;
out.bnd = bnd;
out.interior = int;
out.x = x;
out.nodes = nodes;
out.elements = elements;
out.yNodal = yNodal;
out.pNodal = pNodal;
out.uNodal = uNodal;
out.tri = tri;
out.yFun = yFun;
out.uFun = uFun;
out.metrics = metrics;

savePath = string(p.Results.SaveMat);
if strlength(savePath) > 0
    S = out; %#ok<NASGU>
    save(savePath, 'S');
end

% ------------------------------------------------------------
% lokale Funktionen
% ------------------------------------------------------------
function dl = l_domain_decsg()
R1 = [3; 4; 0; 1; 1; 0; 0; 0; 1; 1]; %[0,1]^2
R2 = [3; 4; 0.5; 1; 1; 0.5; 0.5; 0.5; 1; 1]; %[0.5,1]^2

gd = [R1, R2];
ns = char('R1','R2')';
sf = 'R1-R2';
[dl, ~] = decsg(gd, sf, ns); %Kanten die den Randbeschreiben
end

function val = y_exact(x, y)
val = sin(pi*x) .* sin(pi*y);
end

function val = p_exact(x, y, lambda)
val = -lambda * 2*pi^2 * sin(pi*x) .* sin(pi*y);
end

function yd = y_desired(x, y, lambda)
yd = (1 + 4*lambda*pi^4) * sin(pi*x) .* sin(pi*y);
end

function bnd = boundary_nodes_from_elements(elements)
% Randknoten ueber Randkanten-Erkennung (Kante kommt nur in einem Dreieck vor)
if isempty(elements)
    bnd = [];
    return;
end
T = elements';
E = [T(:, [1, 2]); T(:, [2, 3]); T(:, [3, 1])];
E = sort(E, 2);
[Eu, ~, ic] = unique(E, 'rows');
counts = accumarray(ic, 1);
boundaryEdges = Eu(counts == 1, :);
bnd = unique(boundaryEdges(:));
end

function M = assemble_mass_matrix_p1(nodes, elements)
% L2-Massenmatrix fuer lineare P1-Dreiecke
if isempty(elements)
    M = sparse(0, 0);
    return;
end

T = elements'; % Nt x 3
nNodes = size(nodes, 2);

x = nodes(1, :)';
y = nodes(2, :)';

x1 = x(T(:, 1)); y1 = y(T(:, 1));
x2 = x(T(:, 2)); y2 = y(T(:, 2));
x3 = x(T(:, 3)); y3 = y(T(:, 3));

area = abs((x2 - x1) .* (y3 - y1) - (x3 - x1) .* (y2 - y1)) / 2; %Fläche der Dreieicke
w = area / 12;

I = [T(:,1); T(:,2); T(:,3); T(:,1); T(:,2); T(:,2); T(:,3); T(:,3); T(:,1)];
J = [T(:,1); T(:,2); T(:,3); T(:,2); T(:,1); T(:,3); T(:,2); T(:,1); T(:,3)];
V = [2*w; 2*w; 2*w; 1*w; 1*w; 1*w; 1*w; 1*w; 1*w];

M = sparse(I, J, V, nNodes, nNodes);
end

function [mesh, hmax, info] = calibrate_mesh(model, targetNodes, tolRel, maxIter)
% Passt Hmax iterativ so an, dass die Knotenzahl ~ targetNodes ist.
hmax = 0.20;
hmin = 1e-3;
hmaxMax = 0.60;

info = struct('iter', [], 'hmax', [], 'nNodes', [], 'relErr', []);

for k = 1:maxIter
    mesh = generateMesh(model, 'Hmax', hmax, 'GeometricOrder', 'linear');
    nNodes = size(mesh.Nodes, 2);
    relErr = (nNodes - targetNodes) / targetNodes;

    info.iter(end+1,1) = k; %#ok<AGROW>
    info.hmax(end+1,1) = hmax;
    info.nNodes(end+1,1) = nNodes;
    info.relErr(end+1,1) = relErr;

    if abs(relErr) <= tolRel
        return;
    end

    % Heuristik in 2D: Knotenzahl ~ 1/hmax^2  =>  hmax ~ sqrt(nNodes/target)
    scale = sqrt(nNodes / targetNodes);
    if ~isfinite(scale) || scale <= 0
        scale = 1.0;
    end

    hmax = hmax * scale;
    hmax = min(max(hmax, hmin), hmaxMax);
end
end

function vq = fem_eval(tri, nodalValues, xq, yq)
% Lineare baryzentrische Interpolation; ausserhalb NaN.
xqv = xq(:);
yqv = yq(:);
P = [xqv, yqv];

tid = pointLocation(tri, P);
valid = ~isnan(tid);

vq = nan(size(xqv));
if any(valid)
    bc = cartesianToBarycentric(tri, tid(valid), P(valid, :));
    conn = tri.ConnectivityList(tid(valid), :);
    vals = nodalValues(conn);
    vq(valid) = sum(bc .* vals, 2);
end

vq = reshape(vq, size(xq));
end

function mask = in_L_domain(x, y)
mask = (x <= 0.5) | (y <= 0.5);
end

function metrics = compute_metrics(yFun, uFun, lambda)
% Fehlerstatistiken auf festem Gitter in [0,1]^2, maskiert auf L-Domaene.
n = 301;
xs = linspace(0, 1, n);
ys = linspace(0, 1, n);
[X, Y] = meshgrid(xs, ys);
mask = in_L_domain(X, Y);

Yh = yFun(X, Y);
Uh = uFun(X, Y);

Ye = y_exact(X, Y);
Ue = u_exact(X, Y);

errY = abs(Yh - Ye);
errU = abs(Uh - Ue);

errY = errY(mask);
errU = errU(mask);

metrics = struct();
metrics.gridN = n;
metrics.lambda = lambda;
metrics.errY_mean = mean(errY, 'omitnan');
metrics.errY_max  = max(errY, [], 'omitnan');
metrics.errU_mean = mean(errU, 'omitnan');
metrics.errU_max  = max(errU, [], 'omitnan');
end

function val = u_exact(x, y)
val = 2*pi^2 * sin(pi*x) .* sin(pi*y);
end

end

% Plot driver for kkt_ex1_L_simple

ndof = 2000; % target DOFs per function (≈ mesh nodes)
[yFun, uFun, out] = kkt_ex1_L_simple(ndof);

disp(out.metrics)

% Quick MATLAB plots
figure('Name','Mesh');
pdemesh(out.model);
axis equal tight

gridN = 300;
xs = linspace(0, 1, gridN);
[X, Y] = meshgrid(xs, xs);
mask = (X <= 0.5) | (Y <= 0.5);

% === Auswertung auf konsistentem gridN x gridN Gitter ===
Yh = yFun(X, Y);
Uh = uFun(X, Y);

% Exakte (manufactured) Referenz auf demselben Gitter
Ye = sin(pi*X) .* sin(pi*Y);
Ue = 2*pi^2 * sin(pi*X) .* sin(pi*Y);

% Punktweise Betragsfehler
Yerr = abs(Yh - Ye);
Uerr = abs(Uh - Ue);

% Maskierung auf L-Domäne für ALLE dargestellten Felder
Yh(~mask) = NaN;
Uh(~mask) = NaN;
Ye(~mask) = NaN;
Ue(~mask) = NaN;
Yerr(~mask) = NaN;
Uerr(~mask) = NaN;

% Fehlerstatistiken nur auf Ω_L (NaNs ignorieren)
max_u_error = max(Uerr(mask), [], 'omitnan');
mean_u_error = mean(Uerr(mask), 'omitnan');
max_y_error = max(Yerr(mask), [], 'omitnan');
mean_y_error = mean(Yerr(mask), 'omitnan');

% === Übersichtsfigure (2 Zeilen x 3 Spalten) ===
fig = figure('Name', 'KKT FEM vs. exakt (L-Domäne)');
tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Vollbild öffnen (für Anzeige/Export)
try; fig.WindowState = 'maximized'; catch; set(fig, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]); end

% Colormaps (approx. Matplotlib):
% - coolwarm: blau -> (hell) -> rot  (für Exakt/Approx)
% - viridis: violett -> gelb          (für Fehler)
cmap_coolwarm = coolwarm_map(256);
cmap_viridis  = viridis_map(256);

% Helper: mark excluded upper-right quadrant with a box
add_boundary_box = @(ax) rectangle(ax, 'Position', [0.5, 0.5, 0.5, 0.5], 'EdgeColor', 'k', 'LineWidth', 1.5);

% --- Row 1: u ---
ax1 = nexttile(tl, 1);
imagesc(ax1, xs, xs, Ue);
axis(ax1, 'image');
set(ax1, 'YDir', 'normal');
colormap(ax1, cmap_coolwarm);
colorbar(ax1);
title(ax1, 'u exakt: $2\pi^2\prod_{i=1}^2\sin(\pi x_i)$', 'Interpreter', 'latex');

ax2 = nexttile(tl, 2);
imagesc(ax2, xs, xs, Uh);
axis(ax2, 'image');
set(ax2, 'YDir', 'normal');
colormap(ax2, cmap_coolwarm);
colorbar(ax2);
title(ax2, 'u Approximation (FEM)');
add_boundary_box(ax2);

ax3 = nexttile(tl, 3);
imagesc(ax3, xs, xs, Uerr);
axis(ax3, 'image');
set(ax3, 'YDir', 'normal');
colormap(ax3, cmap_viridis);
colorbar(ax3);
title(ax3, sprintf('u punktweiser Fehler (max=%.2e, mean=%.2e)', max_u_error, mean_u_error));
add_boundary_box(ax3);

% --- Row 2: y ---
ax4 = nexttile(tl, 4);
imagesc(ax4, xs, xs, Ye);
axis(ax4, 'image');
set(ax4, 'YDir', 'normal');
colormap(ax4, cmap_coolwarm);
colorbar(ax4);
title(ax4, 'y exakt: $\prod_{i=1}^2\sin(\pi x_i)$', 'Interpreter', 'latex');

ax5 = nexttile(tl, 5);
imagesc(ax5, xs, xs, Yh);
axis(ax5, 'image');
set(ax5, 'YDir', 'normal');
colormap(ax5, cmap_coolwarm);
colorbar(ax5);
title(ax5, 'y Approximation (FEM)');
add_boundary_box(ax5);

ax6 = nexttile(tl, 6);
imagesc(ax6, xs, xs, Yerr);
axis(ax6, 'image');
set(ax6, 'YDir', 'normal');
colormap(ax6, cmap_viridis);
colorbar(ax6);
title(ax6, sprintf('y punktweiser Fehler (max=%.2e, mean=%.2e)', max_y_error, mean_y_error));
add_boundary_box(ax6);

% Schriftgröße der Subplot-Titel
set([ax1, ax2, ax3, ax4, ax5, ax6], 'TitleFontSizeMultiplier', 1.6*get(ax1, 'TitleFontSizeMultiplier'));

set(get(ax2, 'Title'), 'FontWeight', 'normal');
set(get(ax3, 'Title'), 'FontWeight', 'normal');
set(get(ax5, 'Title'), 'FontWeight', 'normal');
set(get(ax6, 'Title'), 'FontWeight', 'normal');

fig.Color = 'w';
drawnow;

tl.Units = 'pixels';
tlPos = tl.OuterPosition;
tlPos(2) = tlPos(2) - 10;
tlPos(4) = tlPos(4) - 10;
tl.OuterPosition = tlPos;
drawnow;

titles = [get(ax1, 'Title'), get(ax2, 'Title'), get(ax3, 'Title'), get(ax4, 'Title'), get(ax5, 'Title'), get(ax6, 'Title')];
set(titles, 'Units', 'pixels');
for k = 1:numel(titles)
	pos = get(titles(k), 'Position');
	pos(2) = pos(2) + 10;
	set(titles(k), 'Position', pos);
end
outFile = fullfile(pwd, sprintf('Approx_%d.png', ndof));
if exist('exportgraphics', 'file'); exportgraphics(fig, outFile, 'Resolution', 600);
else; print(fig, outFile, '-dpng', '-r600'); end


% --- Local colormap helpers (self-contained) ---
function cmap = viridis_map(n)
%VIRIDIS_MAP  Approximation of Matplotlib 'viridis' (violet -> yellow).
% Uses a small set of anchor colors and linear interpolation.
if nargin < 1
	n = 256;
end
anchors = [
	68,  1, 84;   % #440154
	72, 40,120;   % ~#482878
	62, 74,137;   % ~#3e4a89
	49,104,142;   % ~#31688e
	38,130,142;   % ~#26828e
	31,158,137;   % ~#1f9e89
	53,183,121;   % ~#35b779
   109,205, 89;   % ~#6dcd59
   180,222, 44;   % ~#b4de2c
   253,231, 37;   % #fde725
]/255;
t = linspace(0, 1, size(anchors, 1));
ti = linspace(0, 1, n);
cmap = interp1(t, anchors, ti, 'linear');
cmap = max(0, min(1, cmap));
end

function cmap = coolwarm_map(n)
%COOLWARM_MAP  Approximation of Matplotlib 'coolwarm' (blue -> red).
% Uses a small set of anchor colors and linear interpolation.
if nargin < 1
	n = 256;
end
anchors = [
	 59, 76,192;  % #3b4cc0 (blue)
	114,150,247;  % ~#7296f6
	190,210,253;  % ~#bed2fd
	247,247,247;  % ~#f7f7f7 (near white)
	244,198,180;  % ~#f4c6b4
	220, 92, 69;  % ~#dc5c45
	180,  4, 38;  % #b40426 (red)
]/255;
t = linspace(0, 1, size(anchors, 1));
ti = linspace(0, 1, n);
cmap = interp1(t, anchors, ti, 'linear');
cmap = max(0, min(1, cmap));
end

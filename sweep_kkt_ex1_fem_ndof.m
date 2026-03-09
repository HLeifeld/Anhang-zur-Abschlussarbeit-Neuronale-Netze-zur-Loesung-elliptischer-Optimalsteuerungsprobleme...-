function results = sweep_kkt_ex1_fem_ndof()
%SWEEP_KKT_EX1_FEM_NDOF  Konvergenz-Sweep: KKT-FEM Fehler vs. ndof.
%
%   Analog zu sweep_ex1_fem_ndof.m, aber fuer kkt_ex1_L_simple(ndof).
%   Wertet Fehler von y und u auf festem gridN x gridN Gitter aus,
%   maskiert auf die L-Domaene Ω_L.

% ---------------- Einstellungen ----------------
ndof_values = [200:200:32000]; %[[100:100:1900],[2000:200:9800],[10000:500:32000]] %[81,111,261,371,541,781,921,1341,1401,1981,2051,2661,2911,3441,3921,4321,5081,5301,6381,6391,7561,7851,8841,9461,10221,11221,11701,13131,13281,14961,15191,16741,17401,18621,19761,20601,22271,24931,27741,30701]

 

gridN = 300;
lam = 0.01;

% Export: schreibe einen Run-Ordner (data/ + figs/), der von Python eingelesen werden kann
export_enabled = true;
export_parent = fullfile(fileparts(mfilename('fullpath')), 'Runs');
run_id = datestr(now, 'yyyymmdd_HHMMSS');
export_run_root = fullfile(export_parent, ['kkt_fem_ndof_' run_id]);

% Gitter + Maske (einmalig, konsistent fuer alle Runs)
xs = linspace(0, 1, gridN);
[X, Y] = meshgrid(xs, xs);
maskL = (X <= 0.5) | (Y <= 0.5);

% Exakte (manufactured) Referenzen auf dem Gitter
Y_exact = sin(pi*X) .* sin(pi*Y);
U_exact = 2*pi^2 * sin(pi*X) .* sin(pi*Y);

% Speicher
nRuns = numel(ndof_values);
ndof_total = nan(nRuns, 1);
ndof_free = nan(nRuns, 1);
y_mean = nan(nRuns, 1);
y_max  = nan(nRuns, 1);
u_mean = nan(nRuns, 1);
u_max  = nan(nRuns, 1);

fprintf('Sweep kkt_ex1_L_simple: ndof=%d..%d, gridN=%d\n', ndof_values(1), ndof_values(end), gridN);

for k = 1:nRuns
    ndof = ndof_values(k);
    fprintf('[%2d/%2d] ndof=%d ... ', k, nRuns, ndof);

    try
        [yFun, uFun, out] = kkt_ex1_L_simple(ndof, 'Lambda', lam); %#ok<ASGLU>

        % Tatsächlich erreichte Knotenzahl (gesamt) und freie DOFs (Innenknoten)
        ndof_total(k) = size(out.mesh.Nodes, 2);
        ndof_free(k) = numel(out.interior);

        % Numerische Loesungen auf dem Gitter
        Yh = yFun(X, Y);
        Uh = uFun(X, Y);

        % Punktweiser Betragfehler
        Yerr = abs(Yh - Y_exact);
        Uerr = abs(Uh - U_exact);

        % Einschraenkung auf L-Gebiet
        YerrL = Yerr(maskL);
        UerrL = Uerr(maskL);

        y_mean(k) = mean(YerrL, 'omitnan');
        y_max(k)  = max(YerrL, [], 'omitnan');
        u_mean(k) = mean(UerrL, 'omitnan');
        u_max(k)  = max(UerrL, [], 'omitnan');

        fprintf('nodes=%d, free=%d.  u(max=%.2e, mean=%.2e)  y(max=%.2e, mean=%.2e)\n', ndof_total(k), ndof_free(k), u_max(k), u_mean(k), y_max(k), y_mean(k));

    catch ME
        fprintf('FAILED: %s\n', ME.message);
        continue;
    end
end

% ---------------- Plot ----------------
fig = figure('Name', 'KKT-FEM Fehler vs ndof (Ω_L)');
fig.Position = [100 100 1400 450];

ax = axes(fig);
hold(ax, 'on');
set(ax, 'YScale', 'log');

% X-Achse: freie DOFs (= Innenknoten pro skalarer Funktion)
x = ndof_free;
valid = isfinite(x);
u_mean_plot = u_mean(valid);
u_max_plot  = u_max(valid);
y_mean_plot = y_mean(valid);
y_max_plot  = y_max(valid);

[x, sortIdx] = sort(x(valid));
u_mean_plot = u_mean_plot(sortIdx);
u_max_plot  = u_max_plot(sortIdx);
y_mean_plot = y_mean_plot(sortIdx);
y_max_plot  = y_max_plot(sortIdx);

color_u = [0.20, 0.60, 0.20]; % gruen
color_y = [0.75, 0.20, 0.20]; % rot

semilogy(ax, x, u_mean_plot, '-o',  'LineWidth', 1.5, 'Color', color_u, 'DisplayName', 'u mean');
semilogy(ax, x, u_max_plot,  '--o', 'LineWidth', 1.5, 'Color', color_u, 'DisplayName', 'u max');
semilogy(ax, x, y_mean_plot, '-o',  'LineWidth', 1.5, 'Color', color_y, 'DisplayName', 'y mean');
semilogy(ax, x, y_max_plot,  '--o', 'LineWidth', 1.5, 'Color', color_y, 'DisplayName', 'y max');

% Safety: ensure semilog scaling stays active
set(ax, 'YScale', 'log');

grid(ax, 'on');
xlabel(ax, 'tatsaechliche ndof (freie DOFs = Innenknoten pro skalarer Funktion)');
ylabel(ax, 'Fehler');
title(ax, sprintf('KKT-FEM: Fehler auf Ω_L (gridN=%d, lambda=%.3g)', gridN, lam));
legend(ax, 'Location', 'southwest');

% X-Ticks sichtbar machen (Werte anzeigen)
xticks(ax, unique(x));
xtickangle(ax, 45);

% Feste y-Skalierung (wie im Python-Plot)
ylim(ax, [1e-5, 2e0]);

% ---------------- Output ----------------
results = struct();
results.ndof_requested = ndof_values(:);
results.ndof_total = ndof_total;
results.ndof_free = ndof_free;
results.gridN = gridN;
results.lambda = lam;
results.u_mean = u_mean;
results.u_max = u_max;
results.y_mean = y_mean;
results.y_max = y_max;

% ---------------- Export (JSON + PNG) ----------------
if export_enabled
    try
        if ~exist(export_run_root, 'dir')
            mkdir(export_run_root);
        end
        data_dir = fullfile(export_run_root, 'data');
        figs_dir = fullfile(export_run_root, 'figs');
        if ~exist(data_dir, 'dir')
            mkdir(data_dir);
        end
        if ~exist(figs_dir, 'dir')
            mkdir(figs_dir);
        end

        export_struct = struct();
        export_struct.run_id = run_id;
        export_struct.created_at = datestr(now, 31);
        export_struct.gridN = gridN;
        export_struct.lambda = lam;
        export_struct.ndof_requested = ndof_values(:)';
        export_struct.ndof_total = ndof_total(:)';
        export_struct.ndof_free = ndof_free(:)';
        export_struct.errors = struct();
        export_struct.errors.L = struct();
        export_struct.errors.L.u = struct('mean', u_mean(:)', 'max', u_max(:)');
        export_struct.errors.L.y = struct('mean', y_mean(:)', 'max', y_max(:)');

        json_text = jsonencode(export_struct);
        json_path = fullfile(data_dir, 'sweep_kkt_fem_ndof.json');
        fid = fopen(json_path, 'w');
        if fid < 0
            error('Could not open JSON for writing: %s', json_path);
        end
        fwrite(fid, json_text, 'char');
        fclose(fid);

        fig_path = fullfile(figs_dir, ['sweep_kkt_fem_errors_vs_ndof_' run_id '.png']);
        if exist('exportgraphics', 'file') == 2
            exportgraphics(fig, fig_path, 'Resolution', 150);
        else
            % Fallback for older MATLAB
            try
                saveas(fig, fig_path);
            catch
                print(fig, fig_path, '-dpng', '-r150');
            end
        end

        fprintf('✓ Exported run folder: %s\n', export_run_root);
        fprintf('  - JSON: %s\n', json_path);
        fprintf('  - FIG:  %s\n', fig_path);
    catch ME
        warning('Export failed: %s', ME.message);
    end
end

end

clear
addpath '..\graphic'
addpath '..\utils'


% load exact solution
u_exact_f1 = @(x,y) exp(sin(2*pi*x));
u_exact_f2 = @(x,y) exp(sin(4*pi*x));
u_exact_C = integral(u_exact_f1,0,1)^2;
u_exact = @(x) exp(sin(2*pi*x(:,1))+sin(2*pi*x(:,2))) - u_exact_C;

% Number of runs
N_runs = 25;

% Maxinum cardinality of the index sets

card_I = 2540; % previous case (simulation takes ~5 + 7 + 27 min on
%Simone's Mac with 1 Run)
%card_I = 14000;

BC_type = 'PERIODIC';
tic
n = 2; % dimension

diffusion = @(x) 1 + 0.2 * exp(sin(2*pi*x(:,1)).*sin(2*pi*x(:,2)));
grad_diffusion{1} = @(x) 0.2 * exp(sin(2*pi*x(:,1)).*sin(2*pi*x(:,2))) .* sin(2*pi*x(:,2)) .* cos(2*pi*x(:,1)) *2*pi;
grad_diffusion{2} = @(x) 0.2 * exp(sin(2*pi*x(:,1)).*sin(2*pi*x(:,2))) .* sin(2*pi*x(:,1)) .* cos(2*pi*x(:,2)) *2*pi;
for k = 3 : n
    grad_diffusion{k} = @(x) zeros(size(x(:,1)));
end

m = 2;
I = generate_index_set('HC',n,m);
while size(I,2) <= card_I
    m = m + 1;
    I = generate_index_set('HC',n,m); % index set for Fourier basis
    I(:,(size(I,2)+1)/2) = [];
end
m = m - 1
I = generate_index_set('HC',n,m); % index set for Fourier basis
I(:,(size(I,2)+1)/2) = [];
D = 1/(2*pi)^2 * diag((1./vecnorm(I)).^2);

N = size(I,2); % number of sampling points for full recovery
s_vals = 2.^(2:9);

m_vals = ceil(2*s_vals);

% random grid to measure the errors
N_error = 2*N;
h_int = 1/N_error;
y1_grid = generate_sampling_grid('uniform',n,N_error); 

u_exact_grid_int = u_exact(y1_grid);

N_error = 2*N;
full_uniform_grid = generate_sampling_grid('uniform',n,N_error);
A_full = generate_collocation_matrix(diffusion, grad_diffusion, I, full_uniform_grid, BC_type);
A_full = A_full * D;
f_full = compute_forcing_given_solution(diffusion, u_exact, full_uniform_grid);
x_exact_approach = A_full\f_full;

x_sort = sort(D * x_exact_approach);

i_s = 0;
for s = s_vals
    fprintf('%d ',s)
    i_s = i_s + 1;
    
    % Number of the sampling points
    m = 2*s;

    
    parfor i_run = 1:N_runs
        
        
        random_grid = generate_sampling_grid('uniform',n,m);
        A_CS = generate_collocation_matrix(diffusion, grad_diffusion, I, random_grid, BC_type);
        A_CS = A_CS * D;
        f_CS = compute_forcing_given_solution(diffusion, u_exact, random_grid);
         
        
        norms = sqrt(sum(abs(A_CS).^2,1));
        A_CS1 = A_CS * diag(1./norms);
        
        % CS using womp
        [x_CS1,res,~,stat] = womp_complex(A_CS1, f_CS,ones(size(A_CS1,2),1),0,s,'l0w',[]);

        [x_qcbp,stat] = wqcbp(A_CS,f_CS,ones(size(A_CS,2),1),norm(A_CS*x_exact_approach-f_CS,2),[]);
        x_CS = x_CS1(:,s) ./ norms(:);
        x_backslash = A_CS\f_CS;
        
        x_CS = D * x_CS;
        x_qcbp = D * x_qcbp;
        x_backslash = D * x_backslash;
        
        
        % Compare solution to the exact one
        u_qcbp = @(y_grid) evaluate_solution_given_coefficients(I, x_qcbp, y_grid, BC_type);
        u_CS = @(y_grid) evaluate_solution_given_coefficients(I, x_CS, y_grid, BC_type);
        u_backslash = @(y_grid) evaluate_solution_given_coefficients(I, x_backslash, y_grid, BC_type);
        u_qcbp_grid_int  = u_qcbp(y1_grid);
        u_CS_grid_int  = u_CS(y1_grid);
        u_backslash_grid_int = u_backslash(y1_grid);
        
        % Compute error
        u_L2_norm                        = h_int * norm(u_exact_grid_int(:),2);
        rel_L2_error_backslash(i_s,i_run)     = h_int * norm(real(u_exact_grid_int(:) - u_backslash_grid_int(:)),2) / u_L2_norm;
        rel_L2_error_qcbp(i_s,i_run) = h_int * norm(u_exact_grid_int(:) - u_qcbp_grid_int(:),2) / u_L2_norm;
        rel_L2_error_CS(i_s,i_run)       = h_int * norm(real(u_exact_grid_int(:) - u_CS_grid_int(:)),2) / u_L2_norm;
        
    end
end




i_s = 0;
for s = s_vals
    i_s = i_s +1;
    for i_run = 1:N_runs
        rel_L2_s_term_error1(i_s,i_run) = norm(x_sort(1:(size(x_sort,1)-s)),1)/sqrt(s) / sqrt(h_int * norm(u_exact_grid_int(:),2)^2);
        rel_L2_s_term_error2(i_s,i_run) = norm(x_sort(1:(size(x_sort,1)-s)),2)/sqrt(s) / sqrt(h_int * norm(u_exact_grid_int(:),2)^2);
    end
end

y_data = zeros(size(s_vals,2),N_runs,2);
y_data(:,:,1) = rel_L2_error_CS;
y_data(:,:,2) = rel_L2_error_qcbp;
y_data(:,:,3) = rel_L2_error_backslash;
% y_data(:,:,4) = rel_L2_s_term_error1;
figure(1)
hmean_plot = plot_book_style(s_vals, y_data, 'shaded', 'mean_std_log10');
vert_plot1 = xline(length(I)/2,'Linewidth',2,'LineStyle',':');
legend([hmean_plot vert_plot1],{'CS','QCBP','Backslash','$$m=|\Lambda|$$'},'Interpreter','latex')
set(gca,'YScale','log')
xlabel('s=m/2')
ylabel('Relative L_2 error')
toc
N = length(I);

file_name = "data/D2_card_"+ num2str(N) +".mat";
save(file_name,'s_vals','y_data','N');



n = 8; % dimension

diffusion = @(x) 1 + 0.2 * exp(sin(2*pi*x(:,1)).*sin(2*pi*x(:,2)));
grad_diffusion{1} = @(x) 0.2 * exp(sin(2*pi*x(:,1)).*sin(2*pi*x(:,2))) .* sin(2*pi*x(:,2)) .* cos(2*pi*x(:,1)) *2*pi;
grad_diffusion{2} = @(x) 0.2 * exp(sin(2*pi*x(:,1)).*sin(2*pi*x(:,2))) .* sin(2*pi*x(:,1)) .* cos(2*pi*x(:,2)) *2*pi;
for k = 3 : n
    grad_diffusion{k} = @(x) zeros(size(x(:,1)));
end

m = 2;
I = generate_index_set('HC',n,m);
while size(I,2) <= card_I
    m = m + 1;
    I = generate_index_set('HC',n,m); % index set for Fourier basis
    I(:,(size(I,2)+1)/2) = [];
end
m = m - 1
I = generate_index_set('HC',n,m); % index set for Fourier basis
I(:,(size(I,2)+1)/2) = [];
D = 1/(2*pi)^2 * diag((1./vecnorm(I)).^2);

N = size(I,2); % number of sampling points for full recovery

% random grid to measure the errors
N_error = 2*N;
h_int = 1/N_error;
y1_grid = generate_sampling_grid('uniform',n,N_error); 

u_exact_grid_int = u_exact(y1_grid);

N_error = 2*N;
full_uniform_grid = generate_sampling_grid('uniform',n,N_error);
A_full = generate_collocation_matrix(diffusion, grad_diffusion, I, full_uniform_grid, BC_type);
A_full = A_full * D;
f_full = compute_forcing_given_solution(diffusion, u_exact, full_uniform_grid);
x_exact_approach = A_full\f_full;
x_sort = sort(D * x_exact_approach);

i_s = 0;
for s = s_vals
    fprintf('%d ',s)
    i_s = i_s + 1;
    
    % Number of the sampling points
    m = 2*s;

    
    parfor i_run = 1:N_runs
        
        
        random_grid = generate_sampling_grid('uniform',n,m);
        A_CS = generate_collocation_matrix(diffusion, grad_diffusion, I, random_grid, BC_type);
        A_CS = A_CS * D;
        f_CS = compute_forcing_given_solution(diffusion, u_exact, random_grid);
         
        
        norms = sqrt(sum(abs(A_CS).^2,1));
        A_CS1 = A_CS * diag(1./norms);
        
        % CS using womp
        [x_CS1,res,~,stat] = womp_complex(A_CS1, f_CS,ones(size(A_CS1,2),1),0,s,'l0w',[]);

        [x_qcbp,stat] = wqcbp(A_CS,f_CS,ones(size(A_CS,2),1),norm(A_CS*x_exact_approach-f_CS,2),[]);
        x_CS = x_CS1(:,s) ./ norms(:);
        x_backslash = A_CS\f_CS;
        
        x_CS = D * x_CS;
        x_qcbp = D * x_qcbp;
        x_backslash = D * x_backslash;
        
        
        % Compare solution to the exact one
        u_qcbp = @(y_grid) evaluate_solution_given_coefficients(I, x_qcbp, y_grid, BC_type);
        u_CS = @(y_grid) evaluate_solution_given_coefficients(I, x_CS, y_grid, BC_type);
        u_backslash = @(y_grid) evaluate_solution_given_coefficients(I, x_backslash, y_grid, BC_type);
        u_qcbp_grid_int  = u_qcbp(y1_grid);
        u_CS_grid_int  = u_CS(y1_grid);
        u_backslash_grid_int = u_backslash(y1_grid);
        
        % Compute error
        u_L2_norm                        = h_int * norm(u_exact_grid_int(:),2);
        rel_L2_error_backslash(i_s,i_run)     = h_int * norm(real(u_exact_grid_int(:) - u_backslash_grid_int(:)),2) / u_L2_norm;
        rel_L2_error_qcbp(i_s,i_run) = h_int * norm(u_exact_grid_int(:) - u_qcbp_grid_int(:),2) / u_L2_norm;
        rel_L2_error_CS(i_s,i_run)       = h_int * norm(real(u_exact_grid_int(:) - u_CS_grid_int(:)),2) / u_L2_norm;
        
    end
end




i_s = 0;
for s = s_vals
    i_s = i_s +1;
    for i_run = 1:N_runs
        rel_L2_s_term_error1(i_s,i_run) = norm(x_sort(1:(size(x_sort,1)-s)),1)/sqrt(s) / sqrt(h_int * norm(u_exact_grid_int(:),2)^2);
        rel_L2_s_term_error2(i_s,i_run) = norm(x_sort(1:(size(x_sort,1)-s)),2)/sqrt(s) / sqrt(h_int * norm(u_exact_grid_int(:),2)^2);
    end
end

y_data = zeros(size(s_vals,2),N_runs,2);
y_data(:,:,1) = rel_L2_error_CS;
y_data(:,:,2) = rel_L2_error_qcbp;
y_data(:,:,3) = rel_L2_error_backslash;
% y_data(:,:,4) = rel_L2_s_term_error1;
figure(2)
hmean_plot = plot_book_style(s_vals, y_data, 'shaded', 'mean_std_log10');
vert_plot1 = xline(length(I)/2,'Linewidth',2,'LineStyle',':');
legend([hmean_plot vert_plot1],{'CS','QCBP','Backslash','$$m=|\Lambda|$$'},'Interpreter','latex')
set(gca,'YScale','log')
xlabel('s=m/2')
ylabel('Relative L_2 error')
toc
N = length(I);

file_name = "data/D8_card_"+ num2str(N) +".mat";
save(file_name,'s_vals','y_data','N');



n = 20; % dimension

diffusion = @(x) 1 + 0.2 * exp(sin(2*pi*x(:,1)).*sin(2*pi*x(:,2)));
grad_diffusion{1} = @(x) 0.2 * exp(sin(2*pi*x(:,1)).*sin(2*pi*x(:,2))) .* sin(2*pi*x(:,2)) .* cos(2*pi*x(:,1)) *2*pi;
grad_diffusion{2} = @(x) 0.2 * exp(sin(2*pi*x(:,1)).*sin(2*pi*x(:,2))) .* sin(2*pi*x(:,1)) .* cos(2*pi*x(:,2)) *2*pi;
for k = 3 : n
    grad_diffusion{k} = @(x) zeros(size(x(:,1)));
end

m = 2;
I = generate_index_set('HC',n,m);
while size(I,2) <= card_I
    m = m + 1;
    I = generate_index_set('HC',n,m); % index set for Fourier basis
    I(:,(size(I,2)+1)/2) = [];
end
m = m - 1
I = generate_index_set('HC',n,m); % index set for Fourier basis
I(:,(size(I,2)+1)/2) = [];
D = 1/(2*pi)^2 * diag((1./vecnorm(I)).^2);

N = size(I,2); % number of sampling points for full recovery

% random grid to measure the errors
N_error = 2*N;
h_int = 1/N_error;
y1_grid = generate_sampling_grid('uniform',n,N_error); 

u_exact_grid_int = u_exact(y1_grid);

N_error = 2*N;
full_uniform_grid = generate_sampling_grid('uniform',n,N_error);
A_full = generate_collocation_matrix(diffusion, grad_diffusion, I, full_uniform_grid, BC_type);
A_full = A_full * D;
f_full = compute_forcing_given_solution(diffusion, u_exact, full_uniform_grid);
x_exact_approach = A_full\f_full;
x_sort = sort(D * x_exact_approach);

i_s = 0;
for s = s_vals
    fprintf('%d ',s)
    i_s = i_s + 1;
    
    % Number of the sampling points
    m = 2*s;

    
    parfor i_run = 1:N_runs
        
        
        random_grid = generate_sampling_grid('uniform',n,m);
        A_CS = generate_collocation_matrix(diffusion, grad_diffusion, I, random_grid, BC_type);
        A_CS = A_CS * D;
        f_CS = compute_forcing_given_solution(diffusion, u_exact, random_grid);
         
        
        norms = sqrt(sum(abs(A_CS).^2,1));
        A_CS1 = A_CS * diag(1./norms);
        
        % CS using womp
        [x_CS1,res,~,stat] = womp_complex(A_CS1, f_CS,ones(size(A_CS1,2),1),0,s,'l0w',[]);

        [x_qcbp,stat] = wqcbp(A_CS,f_CS,ones(size(A_CS,2),1),norm(A_CS*x_exact_approach-f_CS,2),[]);
        x_CS = x_CS1(:,s) ./ norms(:);
        x_backslash = A_CS\f_CS;
        
        x_CS = D * x_CS;
        x_qcbp = D * x_qcbp;
        x_backslash = D * x_backslash;
        
        
        % Compare solution to the exact one
        u_qcbp = @(y_grid) evaluate_solution_given_coefficients(I, x_qcbp, y_grid, BC_type);
        u_CS = @(y_grid) evaluate_solution_given_coefficients(I, x_CS, y_grid, BC_type);
        u_backslash = @(y_grid) evaluate_solution_given_coefficients(I, x_backslash, y_grid, BC_type);
        u_qcbp_grid_int  = u_qcbp(y1_grid);
        u_CS_grid_int  = u_CS(y1_grid);
        u_backslash_grid_int = u_backslash(y1_grid);
        
        % Compute error
        u_L2_norm                        = h_int * norm(u_exact_grid_int(:),2);
        rel_L2_error_backslash(i_s,i_run)     = h_int * norm(real(u_exact_grid_int(:) - u_backslash_grid_int(:)),2) / u_L2_norm;
        rel_L2_error_qcbp(i_s,i_run) = h_int * norm(u_exact_grid_int(:) - u_qcbp_grid_int(:),2) / u_L2_norm;
        rel_L2_error_CS(i_s,i_run)       = h_int * norm(real(u_exact_grid_int(:) - u_CS_grid_int(:)),2) / u_L2_norm;
        
    end
end




i_s = 0;
for s = s_vals
    i_s = i_s +1;
    for i_run = 1:N_runs
        rel_L2_s_term_error1(i_s,i_run) = norm(x_sort(1:(size(x_sort,1)-s)),1)/sqrt(s) / sqrt(h_int * norm(u_exact_grid_int(:),2)^2);
        rel_L2_s_term_error2(i_s,i_run) = norm(x_sort(1:(size(x_sort,1)-s)),2)/sqrt(s) / sqrt(h_int * norm(u_exact_grid_int(:),2)^2);
    end
end

y_data = zeros(size(s_vals,2),N_runs,2);
y_data(:,:,1) = rel_L2_error_CS;
y_data(:,:,2) = rel_L2_error_qcbp;
y_data(:,:,3) = rel_L2_error_backslash;
% y_data(:,:,4) = rel_L2_s_term_error1;
figure(3)
hmean_plot = plot_book_style(s_vals, y_data, 'shaded', 'mean_std_log10');
vert_plot1 = xline(length(I)/2,'Linewidth',2,'LineStyle',':');
legend([hmean_plot vert_plot1],{'CS','QCBP','Backslash','$$m=|\Lambda|$$'},'Interpreter','latex')
set(gca,'YScale','log')
xlabel('s=m/2')
ylabel('Relative L_2 error')
toc
N = length(I);

file_name = "data/D20_card_"+ num2str(N) +".mat";
save(file_name,'s_vals','y_data','N');
clear
addpath '..\graphic'
addpath '..\utils'
tic;
% Non sparse diffusion function
grad_diffusion{1} = @(x) 0.2 * exp(sin(2*pi*x(:,1)).*sin(2*pi*x(:,2))) .* sin(2*pi*x(:,2)) .* cos(2*pi*x(:,1)) *2*pi;
grad_diffusion{2} = @(x) 0.2 * exp(sin(2*pi*x(:,1)).*sin(2*pi*x(:,2))) .* sin(2*pi*x(:,1)) .* cos(2*pi*x(:,2)) *2*pi;
diffusion = @(x) 1 + 0.2 * exp(sin(2*pi*x(:,1)).*sin(2*pi*x(:,2)));


n = 2; % dimension
BC_type = 'PERIODIC'; % Type of boundary condition

I = generate_index_set('HC',n,26); % index set for Fourier basis
I(:,(size(I,2)+1)/2) = []; 

N = size(I,2); % number of sampling points for full recovery

s_vals = 14:2:80;
y_data = zeros(size(s_vals,2),1,3);
m_vals = ceil(2*s_vals);
N_runs = 25;

for sp = 1:3
    
    u_exact = @(x) zeros(size(x(:,1)));
    sparsity = sp*4;
    rand_freq = randperm(16,sparsity);
    for i = 1 : sparsity
        m = mod(rand_freq(i),4)+1;
        k = floor((rand_freq(i)-1)/4)+1;
        c = rand(1);
        u_exact = @(x)  u_exact(x) + c * sin(2*pi * m * x(:,1)) .* sin(2*pi * k * x(:,2));
    end
    % random grid to measure the errors
    N_error = 200;
    h_int = 1/N_error;
    y1_grid = generate_sampling_grid('uniform',n,N_error); 

    u_exact_grid_int = u_exact(y1_grid);

    i_s = 0;
    for s = s_vals
        fprintf('%d ',s)
        i_s = i_s + 1;

        % Number of the sampling points
        m = 2*s;


        parfor i_run = 1:N_runs


            random_grid = generate_sampling_grid('uniform',n,m);
            A_CS = generate_collocation_matrix(diffusion, grad_diffusion, I, random_grid, BC_type);
            f_CS = compute_forcing_given_solution(diffusion, u_exact, random_grid);


            norms = sqrt(sum(abs(A_CS).^2,1));
            A_CS1 = A_CS  ./norms;

            % CS using womp
            [x_CS1,res,~,stat] = womp_complex(A_CS1, f_CS,ones(size(A_CS,2),1),0,s,'l0w',[]);

            x_CS = x_CS1(:,s) ./ norms(:);

            % Compare solution to the exact one
            u_CS = @(y_grid) evaluate_solution_given_coefficients(I, x_CS, y_grid, BC_type);
            u_CS_grid_int  = u_CS(y1_grid);

            % Compute error
            u_L2_norm                        = h_int * norm(u_exact_grid_int(:),2);
            rel_L2_error_CS(i_s,i_run)       = h_int * norm(real(u_exact_grid_int(:) - u_CS_grid_int(:)),2) / u_L2_norm;

        end
        y_data(i_s,1,sp) = nnz(rel_L2_error_CS(i_s,:)<1e-6)/N_runs;
    end
end


figure(6)
hmean_plot = plot_book_style(s_vals, y_data, 'shaded', 'mean_std_log10');
legend(hmean_plot,{'q=4','q=8','q=12'})
xlabel('m')
ylabel('Success rate')
N = length(I);

toc
save('data/D2_EtaNonSparse_Success_rate.mat','s_vals','y_data','N');
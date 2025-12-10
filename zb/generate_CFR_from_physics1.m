function [Y_noisy_CFR_cell, H_true_CFR_cell, true_delays_cell, f_subcarrier_new] = generate_CFR_from_physics1(targetTrajectory, dataVA, parameters, model_params)
%
% generate_CFR_from_physics.m
% 
% 功能描述:
%   1. 基于物理轨迹生成多径时延和增益。
%   2. 自动计算无混叠的子载波间隔 f_new。
%   3. 构建信道频率响应 (CFR)。
%   4. [关键] 在特定时间窗 (400-600步) 注入恶劣的物理干扰 (衰减+偏差+高噪)，
%      用于验证自适应 SLAM 算法的鲁棒性。
%

% --- 1. 初始化参数 ---
c = 3.0e8; 

% 从输入结构体获取参数
numSteps = model_params.numSteps; 
N_subcarrier = model_params.N_subcarrier;
f_carrier = model_params.f_s;
snr_db = model_params.snr;

m_indices = (0:N_subcarrier-1).';
numSensors = length(dataVA); 

% 初始化存储变量
true_delays_cell = cell(numSteps, numSensors);
true_amps_cell   = cell(numSteps, numSensors); % 存储未受干扰的原始增益
global_max_tau   = 0; 

% 用于计算基准噪声水平的累加器
sum_power_normal = 0;
count_samples_normal = 0;

% =========================================================================
% 第一阶段: 扫描路径，计算 tau_max，并统计正常信号功率
% =========================================================================
fprintf('--- [Physics] 正在扫描路径以确定频率参数和基准功率 ---\n');

for step = 1:numSteps
    target_pos = targetTrajectory(:, step); 
    
    for sensor = 1:numSensors
        positions = dataVA{sensor}.positions;
        visibility = dataVA{sensor}.visibility;
        [~, numAnchors] = size(positions);

        true_delays_sensor = [];
        true_amps_sensor = [];

        for anchor = 1:numAnchors
            if(visibility(anchor, step)) 
                % 1. 几何计算
                anchor_pos = positions(:, anchor);
                dist = sqrt((anchor_pos(1) - target_pos(1)).^2 + (anchor_pos(2) - target_pos(2)).^2);
                tau_k = dist / c; 
                
                % 2. 物理增益 (Free Space Path Loss + Rayleigh Fading)
                sigma_k_sq = c / (4 * pi * dist * f_carrier); 
                % 随机复高斯增益
                alpha_k = sqrt(sigma_k_sq/2) * (randn() + 1i * randn()); 

                % 3. 存储
                true_delays_sensor = [true_delays_sensor; tau_k];
                true_amps_sensor = [true_amps_sensor; alpha_k];
                
                % 4. 更新全局最大时延
                if tau_k > global_max_tau
                    global_max_tau = tau_k;
                end
                
                % 5. 统计正常信号功率 (用于后续生成固定的底噪)
                % 我们只统计前 200 步作为"洁净"样本
                if step < 200
                    sum_power_normal = sum_power_normal + abs(alpha_k)^2;
                    count_samples_normal = count_samples_normal + 1;
                end
            end
        end
        
        true_delays_cell{step, sensor} = true_delays_sensor;
        true_amps_cell{step, sensor} = true_amps_sensor;
    end
end

% 计算基准平均功率 (Reference Signal Power)
if count_samples_normal > 0
    avg_power_ref = sum_power_normal / count_samples_normal;
else
    avg_power_ref = 1e-6; % Fallback
end

% =========================================================================
% 第二阶段: 设定频率参数
% =========================================================================
% 设定 f_new 以避免混叠 (Nyquist: f <= 1/2*tau_max)
% 留一点余量，除以 2.1
f_subcarrier_new = 1 / (2.1 * global_max_tau);

fprintf('--- [Info] 最大时延: %.2e s, 设定子载波间隔: %.2e Hz ---\n', global_max_tau, f_subcarrier_new);
fprintf('--- [Info] 基准信号功率: %.2e ---\n', avg_power_ref);

% =========================================================================
% 第三阶段: 生成 CFR 并注入干扰
% =========================================================================
Y_noisy_CFR_cell = cell(numSteps, numSensors);
H_true_CFR_cell  = cell(numSteps, numSensors);

for step = 1:numSteps
    
    % --- 定义是否处于"恶劣环境" (Bad Condition) ---
    % 在 400 到 600 步之间触发
    is_bad_condition = (step >= 400 && step <= 600);
    
    for sensor = 1:numSensors
        true_delays_sensor = true_delays_cell{step, sensor};
        true_amps_sensor = true_amps_cell{step, sensor}; % 这是原始的、健康的增益
        K_paths = length(true_delays_sensor);
        
        % 初始化 CFR
        H_true_CFR_sensor = zeros(N_subcarrier, 1);
        
        % -----------------------------------------------------------------
        % A. 处理路径增益与偏差
        % -----------------------------------------------------------------
        current_alphas = true_amps_sensor;
        
        if is_bad_condition
            % 1. 信号大幅衰减 (模拟遮挡)
            % 衰减 20 倍 (-26dB)，让自适应算法检测到 amps 变小
            current_alphas = current_alphas * 0.05; 
            
            % 2. 注入系统性偏差 (Bias)
            % 这是一个 0.3m 的恒定距离漂移，用于测试算法是否会发散
            bias_dist = 0.3; 
            bias_tau = bias_dist / c;
            % 构造频域线性相位偏移
            bias_phasor = exp(-1i * 2 * pi * f_subcarrier_new * bias_tau * m_indices);
        else
            bias_phasor = ones(N_subcarrier, 1);
        end
        
        % -----------------------------------------------------------------
        % B. 构建无噪 CFR (H_true)
        % -----------------------------------------------------------------
        for k = 1:K_paths
            tau_k = true_delays_sensor(k);
            alpha_k = current_alphas(k); % 使用可能衰减过的增益
            
            % 计算相位项
            s_tau_k = 2 * pi * f_subcarrier_new * tau_k;
            
            % 叠加路径
            path_signal = alpha_k * exp(-1i * m_indices * s_tau_k);
            
            % 累加到总 CFR
            H_true_CFR_sensor = H_true_CFR_sensor + path_signal;
        end
        
        % 应用系统偏差 (如果是恶劣环境)
        % 这会让所有路径的相位整体旋转，等效于测距整体偏大 0.3m
        if is_bad_condition
            H_true_CFR_sensor = H_true_CFR_sensor .* bias_phasor;
        end
        
        % -----------------------------------------------------------------
        % C. 生成噪声 (关键步骤)
        % -----------------------------------------------------------------
        % 策略：
        % - 正常情况：基于当前信号功率和设定的 SNR 生成噪声。
        % - 恶劣情况：锁定噪声底限！使用"未衰减的基准功率"来生成噪声，
        %   甚至进一步放大噪声。这样相对于衰减后的信号，SNR 会极低。
        
        if is_bad_condition
            % 计算基于"正常信号"的噪声标准差
            % 并额外放大 2 倍，制造剧烈的测量跳变
            Pn_target = avg_power_ref * 10^(-snr_db/10); 
            noise_std = sqrt(0.5 * Pn_target) * 2.0; 
        else
            % 正常 SNR 逻辑
            % 基于当前 H 的功率
            Pn_current = mean(abs(H_true_CFR_sensor).^2);
            noise_std = sqrt(0.5 * Pn_current * 10^(-snr_db/10));
        end
        
        % 生成复高斯白噪声
        eps = noise_std * (randn(N_subcarrier,1) + 1i * randn(N_subcarrier,1));
        
        % 最终的观测信号
        Y_noisy_CFR_sensor = H_true_CFR_sensor + eps;

        % 存储
        Y_noisy_CFR_cell{step, sensor} = Y_noisy_CFR_sensor;
        H_true_CFR_cell{step, sensor}  = H_true_CFR_sensor;
        
    end
end

end
##-----------------------------------------------------------------
# Gráfico final — Kaplan-Meier colorido por grupo + pontos do modelo
##-----------------------------------------------------------------

# Kaplan–Meier por grupo
mKM <- survfit(Surv(y, status) ~ x1 + x2,data=dados, se.fit = FALSE)

# Cores bem distintas para os 4 grupos (sexo × idade)
cores_km <- c("#0072B2", "#E69F00", "#009E73", "#D55E00")  
# Azul, laranja, verde e vermelho queimado — padrão ggplot2

# Nomes dos grupos
grupos_fator <- interaction(dados$x1, dados$x2)
grupos_nomes <- levels(grupos_fator)

# Plot Kaplan-Meier com cores por grupo
plot(mKM,
     col = cores_km,
     lwd = 2.5,
     lty = 1,
     xlab = "Tempo (meses)",
     ylab = "Função de sobrevivência",
     main = "Kaplan-Meier vs Modelo por Grupo",
     cex.axis = 1.3,
     cex.lab = 1.5,
     ylim = c(0, 1))

# Criar um data.frame com os valores do modelo
dados_plot <- data.frame(y = y,
                         Spop = Spop,
                         Grupo = grupos_fator)

# Adiciona os pontos coloridos do modelo, por grupo
for (i in seq_along(grupos_nomes)) {
    grupo_atual <- grupos_nomes[i]
    dados_grupo <- dados_plot[dados_plot$Grupo == grupo_atual, ]
    
    points(dados_grupo$y,
           dados_grupo$Spop,
           pch = 16,          # ponto redondo sólido
           col = cores_km[i], # cor do grupo
           cex = 0.8)
}

# Legenda principal (curvas KM + modelo)
legend("bottomright",
       legend = grupos_nomes,
       col = cores_km,
       lty = 1,
       lwd = 2.5,
       pch = 16,
       pt.cex = 1.2,
       cex = 1.1,
       bty = "n",
       y.intersp = 1.1,
       inset = c(0.02, 0.05))

# Grade leve
grid(col = "gray85", lty = "dotted")

##-----------------------------------------------------------------
# Gráfico final — Kaplan-Meier colorido por grupo + pontos do modelo
##-----------------------------------------------------------------

# Kaplan–Meier por grupo (calcule se necessário)
mKM <- survfit(Surv(y, status) ~ x1 + x2, se.fit = FALSE)

# Cores desejadas (escolha 4 cores)
cores_km_base <- c("#0072B2", "#E69F00", "#009E73", "#D55E00")  

# Recupera os nomes das estratos exatamente como o survfit produziu
strata_names <- names(mKM$strata)
# Exemplo de strata_names: "x1=(<65), x2=(Feminino) " ...

# Se for necessário, remova espaços extras nos nomes (opcional)
strata_names <- trimws(strata_names)

# Cria um vetor nomeado de cores na mesma ordem das estratos
cores_km <- setNames(cores_km_base[seq_along(strata_names)], strata_names)

# Verifique os níveis do interaction usado nos dados
grupos_fator <- interaction(dados$x1, dados$x2)
# Ajusta os níveis para terem o mesmo formato/ordem das 'strata_names'
# Criamos uma versão nomeada dos níveis para comparar
nivels <- trimws(levels(grupos_fator))

# Se os nomes dos níveis diferirem em formatação dos nomes das estratas,
# transforme os níveis para o mesmo formato de 'strata_names'.
# Uma estratégia robusta: comparar por presença das partes (x1=(...), x2=(...))
# Vamos criar uma função auxiliar para mapear cada combinação ao nome de strata:
make_strata_label <- function(f) {
  # f é algo como "(<65).(Feminino)" dependendo de interaction default
  parts <- strsplit(as.character(f), "\\.")[[1]]
  # parts[1] corresponde a x1; parts[2] a x2
  paste0("x1=", parts[1], ", x2=", parts[2])
}

# Mapeia cada observação para o nome de strata do survfit
mapped_labels <- sapply(as.character(grupos_fator), make_strata_label)
mapped_labels <- trimws(mapped_labels)

# Plot Kaplan-Meier com cores por strata (mantendo a ordem do survfit)
plot(mKM,
     col = cores_km,     # cores nomeadas (survfit usa names)
     lwd = 2.5,
     lty = 1,
     xlab = "Tempo (meses)",
     ylab = "Função de sobrevivência",
     main = "Kaplan-Meier vs Modelo por Grupo",
     cex.axis = 1.3,
     cex.lab = 1.5,
     ylim = c(0, 1))

# Data.frame para plotar pontos do modelo
dados_plot <- data.frame(y = y,
                         Spop = Spop,
                         StrataLabel = mapped_labels,
                         Grupo = as.character(grupos_fator),
                         stringsAsFactors = FALSE)

# Adiciona pontos coloridos do modelo, usando a cor mapeada corretamente
for (lab in unique(dados_plot$StrataLabel)) {
    sel <- dados_plot$StrataLabel == lab
    points(dados_plot$y[sel],
           dados_plot$Spop[sel],
           pch = 16,
           col = cores_km[lab],  # usa cor pelo nome da strata
           cex = 0.8)
}

# Legenda (usa a mesma ordem de strata do survfit)
legend("bottomright",
       legend = strata_names,
       col = cores_km,
       lty = 1,
       lwd = 2.5,
       pch = 16,
       pt.cex = 1.2,
       cex = 1.1,
       bty = "n",
       y.intersp = 1.1,
       inset = c(0.02, 0.05))

grid(col = "gray85", lty = "dotted")


########

# -----------------------------------------------------------------
# 1. Definir os valores de p0 correlacionados aos grupos
# -----------------------------------------------------------------
# Certifique-se de que a ordem aqui segue a ordem dos níveis de 'interaction(x1, x2)'
# No seu caso: (<65).Feminino, (<65).Masculino, (>=65).Feminino, (>=65).Masculino
p0_valores <- c("p0=0.8429", "p0=0.8938", "p0=0.6127", "p0=0.7128")

# -----------------------------------------------------------------
# 2. Criar nomes de legenda que combinam Grupo + p0
# -----------------------------------------------------------------
# Vamos criar um vetor de nomes formatados
legenda_com_p0 <- paste(strata_names, "|", p0_valores)

# -----------------------------------------------------------------
# 3. Plotagem
# -----------------------------------------------------------------
plot(mKM,
     col = cores_km,
     lwd = 2.5,
     lty = 1,
     xlab = "Tempo (meses)",
     ylab = "Função de sobrevivência",
     main = "Kaplan-Meier vs Modelo (com Prob. de Cura)",
     cex.axis = 1.3,
     cex.lab = 1.5,
     ylim = c(0, 1))

# Adiciona os pontos do modelo (usando o loop que já tínhamos)
for (lab in unique(dados_plot$StrataLabel)) {
    sel <- dados_plot$StrataLabel == lab
    points(dados_plot$y[sel],
           dados_plot$Spop[sel],
           pch = 16,
           col = cores_km[lab],
           cex = 0.8)
}

# -----------------------------------------------------------------
# 4. Legenda Atualizada
# -----------------------------------------------------------------
legend("bottomleft",          # Mudei para bottomleft se houver espaço, ou mantenha bottomright
       legend = legenda_com_p0,
       col = cores_km,
       lty = 1,
       lwd = 2.5,
       pch = 16,
       pt.cex = 1.2,
       cex = 0.9,             # Diminuí levemente para caber o texto extra
       bty = "n",
       y.intersp = 1.3,       # Aumenta o espaço vertical entre linhas
       inset = c(0.02, 0.05),
       title = "Grupos e Probabilidade de Cura (p0)")

grid(col = "gray85", lty = "dotted")



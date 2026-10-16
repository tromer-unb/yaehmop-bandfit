# Guia prático em português — do CIF e PNG às bandas YAeHMOP

**Para quem tem um CIF, uma figura de bandas e quer ajustar e reproduzir o espectro de um material.** Este é o fluxo individual para semicondutores não magnéticos com gap positivo e referências de 2, 4, 6 ou 8 bandas. O exemplo abaixo usa o nome MoS2, mas o **CIF e o PNG são fornecidos pelo leitor**. Não é uma reprodução automática de DFT a partir do CIF.

## 1. Instalar e colocar os arquivos

~~~bash
git clone https://github.com/tromer-unb/yaehmop-bandfit.git
cd yaehmop-bandfit

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

mkdir -p inputs/MoS2
cp /caminho/do/arquivo/structure.cif inputs/MoS2/structure.cif
cp /caminho/da/imagem/bands.png inputs/MoS2/bands.png
cp examples/configs/material_2bands_template.ini inputs/MoS2/file.ini
~~~

Edite **inputs/MoS2/file.ini**. Ele deve conter os limites de energia da sua imagem (**emin**, **emax**), **a trajetória real no espaço recíproco** do seu cálculo DFT, os rótulos e **n_pairs**. O template usa Γ–M–K–Γ e −6/+6 eV **como exemplo, não como informação determinada automaticamente de todo CIF**.

Para calcular espectros EHT de verdade, instale o **YAeHMOP original** separadamente e indique seus próprios caminhos:

~~~bash
export YAEHMOP_BIND=/caminho/para/yaehmop/bind
export YAEHMOP_PARAM_FILE=/caminho/para/yaehmop/eht_parms.dat
~~~

## 2. Conferir e padronizar o PNG (sem fazer fitting)

~~~bash
python USER_WORKFLOW/inspect_png.py inputs/MoS2/bands.png \
  --outdir outputs/inspecao

python USER_WORKFLOW/bands_fit.py \
  inputs/MoS2/structure.cif inputs/MoS2/file.ini \
  --prepare-only --outdir outputs/MoS2
~~~

**Abra outputs/MoS2/reference_prepared/extraction_preview.png.** Verifique visualmente se as curvas extraídas correspondem às curvas e aos cruzamentos visíveis da fonte.

Se o programa não identificar as cores, informe o retângulo e os tons reais do gráfico:

~~~bash
python USER_WORKFLOW/prepare_band_png.py inputs/MoS2/bands.png \
  --mode colors --n-pairs 1 \
  --emin -6 --emax 6 \
  --bbox 110,80,920,650 \
  --colors '#c82d37,#1c7ad5' \
  --outdir outputs/MoS2_png
~~~

**Os números de bbox e as cores acima são ilustrativos.** O comando cria **bands_2_layers.png**, **extracted_curves.csv**, **extraction_preview.png** e **extraction_qa.json**. O PNG em camadas é padronizado e codifica bandas diferentes em bits diferentes do mesmo pixel. **Ele não recupera linhas que estão totalmente ocultas na imagem.**

## 3. Fazer o ajuste individual (executa o YAeHMOP)

~~~bash
python USER_WORKFLOW/bands_fit.py \
  inputs/MoS2/structure.cif inputs/MoS2/file.ini \
  --pairs 1 --budget 400 \
  --outdir outputs/MoS2
~~~

Resultados:

| Arquivo | O que fornece |
|---|---|
| outputs/MoS2/bands_comparison.png | Curvas DFT e EHT sobrepostas |
| outputs/MoS2/bands_comparison.pdf | Mesma comparação em PDF |
| outputs/MoS2/bands_comparison.csv | Bandas numéricas e suas diferenças |
| outputs/MoS2/fit/param_best_native.dat | Tabela atômica nativa ajustada |
| outputs/MoS2/fit/fit.json | Otimização e parâmetros incluindo K |
| outputs/MoS2/run_summary.json | RMSE, K, configurações e procedência |

**--pairs 1 = 1V+1C**, **--pairs 2 = 2V+2C**, **--pairs 3 = 3V+3C**, **--pairs 4 = 4V+4C**. Para aumentar o número de bandas, a referência precisa conter esse número de curvas verdadeiras.

## 4. Reproduzir TODAS as bandas do Hamiltoniano físico, sem reajustar

~~~bash
python USER_WORKFLOW/bands_fit.py \
  inputs/MoS2/structure.cif inputs/MoS2/file.ini \
  --from-fit outputs/MoS2/fit/fit.json \
  --outdir outputs/MoS2_full
~~~

O comando gera **bands_full.png**, **bands_full.pdf**, **bands_full.csv** e **bands_full.npz**. CSV/NPZ incluem todos os autovalores orbitais físicos calculados; **as bandas não selecionadas para fitting não foram otimizadas diretamente**.

## 5. Quando usar CSV em vez do PNG?

Se as curvas do PNG estão sobrepostas, incompletas ou da mesma cor, prefira energias tabuladas. Um exemplo de **estrutura** de arquivo:

~~~csv
x,band_1,band_2
0.00,-1.20,1.80
0.05,-1.25,1.75
0.10,-1.31,1.70
~~~

O arquivo completo precisa de ao menos 15 linhas, com x crescente de 0 até 1 e bandas ordenadas por energia. Salve como **inputs/MoS2/bands.csv** e execute:

~~~bash
python USER_WORKFLOW/bands_fit.py \
  inputs/MoS2/structure.cif inputs/MoS2/file.ini \
  --image inputs/MoS2/bands.csv --mode csv \
  --pairs 1 --budget 400 \
  --outdir outputs/MoS2_csv
~~~

No GitHub atual, **--image e --png são aliases da mesma opção**.

## 6. Transferibilidade e casos especiais

Para **apenas visualizar** experimentos arquivados de transferência, sem executar DFT ou fitting:

~~~bash
python3 quick.py chemical-transfer
python3 quick.py transfer
~~~

O programa **parameter_fitter/fit_shared.py** aceita YAML com vários sistemas de treinamento e teste, mas **ainda não oferece o mesmo suporte direto a PNGs com apenas 2/4/6/8 bandas** do programa individual. Não basta usar duas imagens comuns para garantir um ajuste compartilhado correto. Veja o [guia específico](SHARED_FITTING.md).

O ajuste por imagem exige bandas e gap identificáveis, uma base orbital suportada e um caminho k correto. Grafeno com cone de Dirac, metais, sistemas magnéticos e SOC precisam de tratamento especializado. O patch de dupla-zeta **2p do carbono** é experimental, separado do executável original e não valida automaticamente bases duplas de outros elementos.

**Mais detalhes:** [guia completo em inglês](FIT_YOUR_MATERIAL.md), [entrada de imagens](IMAGE_INPUT.md), [flags](CLI_REFERENCE.md) e [instalação](INSTALLATION.md).

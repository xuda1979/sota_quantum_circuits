#!/usr/bin/env python3
"""Generate the detailed paper-replication report as a .docx (Chinese).

Run:  python3 gen_replication_docx.py [out.docx]
Data sources: REPLICATION_REPORT.md (2026-11-13 full replication, 17/17
entries) plus the live re-verification spot checks run on 2026-11-17.
"""

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
GREEN = RGBColor(0x2E, 0x7D, 0x32)

# ----------------------------------------------------------------------------
# data (verified numbers; see REPLICATION_REPORT.md and 2026-11-17 spot checks)
# ----------------------------------------------------------------------------

TABLE1 = [
    # task, circuit, paper (T,Q), certified (T,Q), this run (T,Q)
    ("dqi-reed-solomon", "发现电路 (Berlekamp–Massey)", "4.4×10⁵, 1,837", "442,397 × 1,837", "442,397 × 1,837"),
    ("dqi-reed-solomon", "khattar-eea (基线)", "5.1×10⁶, 1,938", "5,125,070 × 1,938", "5,125,070 × 1,938"),
    ("dqi-hermitian", "发现电路 (Lanczos)", "9.3×10⁵, 2,395", "926,912 × 2,395", "926,912 × 2,395"),
    ("dqi-hermitian", "sakata-fengrao (基线)", "1.0×10⁸, 1.9×10⁵", "104,263,342 × 194,053", "104,263,342 × 194,053"),
    ("hhl-angle", "发现电路 (dyadic quadratic)", "1,348, 115", "1,348 × 115", "1,348 × 115"),
    ("hhl-angle", "haner-piecewise (基线)", "1.6×10⁴, 300", "16,468 × 300", "16,468 × 300"),
    ("coulomb-potential", "发现电路 (line table)", "1,140, 142", "1,140 × 142", "1,140 × 142"),
    ("coulomb-potential", "haner-piecewise (基线)", "6,355, 252", "6,355 × 252", "6,355 × 252"),
    ("gaussian", "发现电路 (pruned QROM)", "1,583, 139", "1,583 × 139", "1,583 × 139"),
    ("gaussian", "haner-piecewise (基线)", "2.4×10⁴, 362", "23,784 × 362", "23,784 × 362"),
    ("chemistry-fe2s2", "发现电路 (frame-block)", "1.5×10⁴, 126", "14,978 × 126", "14,978 × 126"),
    ("chemistry-fe2s2", "babbush-composed (基线)", "5.5×10⁴, 139", "55,473 × 139", "55,473 × 139"),
    ("lattice-gauge-plaquette", "发现电路 (specialized rep.)", "1.7×10⁵, 1,859", "172,128 × 1,859", "172,128 × 1,859"),
    ("lattice-gauge-plaquette", "osorio-composed (基线)", "1.3×10⁶, 1,927", "1,271,424 × 1,927", "1,271,424 × 1,927"),
    ("modinv-point-add", "divsteps (发现电路)", "4.0×10⁶, 935", "4,006,673 × 935", "4,006,673 × 935"),
    ("modinv-point-add", "euclidean (发现电路, SI II.2)", "2.1×10⁷, 843", "20,521,949 × 843", "20,521,949 × 843"),
    ("modinv-point-add", "luo-eea (基线)", "7.0×10⁷, 838", "69,529,757 × 838", "69,529,757 × 838"),
]

GAINS = [
    ("Reed-Solomon DQI 解码器", "9,932,385,660", "812,683,289", "12", "12.2"),
    ("HHL 旋转角", "4,940,400", "155,020", "32", "31.9"),
    ("库仑势", "1,601,460", "161,880", "9.9", "9.9"),
    ("定点高斯函数", "8,609,808", "220,037", "39", "39.1"),
    ("Fe2S2 别名-Select", "7,710,747", "1,887,228", "4.1", "4.1"),
    ("非阿贝尔格点规范", "2,450,034,048", "319,985,952", "7.7", "7.7"),
]

DIVSTEP = [
    ("wide.py", "3,293,278", "1,191", "3,922,294,098"),
    ("fill.py", "3,637,655", "1,031", "3,750,422,305"),
    ("circuit.py (j_store=5)", "4,006,673", "935", "3,746,239,255"),
    ("j5.py", "4,317,926", "927", "4,002,717,402"),
    ("j6.py", "4,427,338", "913", "4,042,159,594"),
    ("j7.py", "4,649,514", "904", "4,203,160,656"),
    ("j8.py", "5,331,729", "897", "4,782,560,913"),
    ("j9.py", "7,866,779", "893", "7,025,033,647"),
]

FAILURES = [
    ("Reed-Solomon 发现电路:10 万次测试中错误符号数", "7", "7"),
    ("Hermitian 发现电路:10 万次测试中错误符号数", "30", "30"),
    ("Divsteps 点加:10 万次测试中失败数", "4", "4"),
    ("Euclidean 点加:10 万次测试中失败数", "4", "4"),
    ("全部基线与数值电路", "0 失败", "0 失败"),
]

LIVE = [
    ("hhl-angle 发现电路", "OK", "1,348", "115", "155,020", "精确一致"),
    ("hhl-angle haner-piecewise 基线", "OK", "16,468", "300", "4,940,400", "精确一致"),
    ("gaussian 发现电路", "OK", "1,583", "139", "220,037", "精确一致"),
    ("gaussian haner-piecewise 基线", "OK", "23,784", "362", "8,609,808", "精确一致"),
    ("coulomb-potential 发现电路", "OK", "1,140", "142", "161,880", "精确一致"),
    ("coulomb-potential haner-piecewise 基线", "OK", "6,355", "252", "1,601,460", "精确一致"),
    ("dqi-reed-solomon 发现电路", "OK", "442,397", "1,837", "812,683,289", "精确一致 (含 7/100,000 错符号)"),
]

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def set_cjk(style_or_run, east="SimSun", latin="Calibri"):
    f = style_or_run.font
    f.name = latin
    rpr = f.element.get_or_add_rPr() if hasattr(f, "element") else None
    if rpr is None:
        # run font
        rpr = f.element.rPr if f.element.rPr is not None else None
    el = f.element if hasattr(f, "element") else None
    # unify: use the underlying element robustly
    try:
        rPr = f.element.rPr
        assert rPr is not None
    except Exception:
        rPr = None
    if rPr is not None:
        rFonts = rPr.find(qn("w:rFonts"))
        if rFonts is None:
            from docx.oxml import OxmlElement
            rFonts = OxmlElement("w:rFonts")
            rPr.append(rFonts)
        rFonts.set(qn("w:eastAsia"), east)


def add_page_number(doc):
    footer = doc.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    from docx.oxml import OxmlElement
    run = p.add_run()
    fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
    run._r.append(fld1); run._r.append(instr); run._r.append(fld2)


def style_doc(doc):
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
    for h, sz, col in (("Heading 1", 16, ACCENT), ("Heading 2", 13, ACCENT), ("Heading 3", 11.5, ACCENT)):
        s = doc.styles[h]
        s.font.name = "Calibri"
        s.font.size = Pt(sz)
        s.font.color.rgb = col
        s.font.bold = True
        s.element.get_or_add_rPr()
        rF = s.element.rPr.find(qn("w:rFonts"))
        if rF is None:
            from docx.oxml import OxmlElement
            rF = OxmlElement("w:rFonts"); s.element.rPr.append(rF)
        rF.set(qn("w:eastAsia"), "SimHei")


def para(doc, text, size=10.5, bold=False, color=None, align=None, space_after=6):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.bold = bold
    if color:
        r.font.color.rgb = color
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p


def bullet(doc, text, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    r.font.size = Pt(size)
    return p


def make_table(doc, headers, rows, widths=None, font=9, header_font=9):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        r = hdr[i].paragraphs[0].add_run(h)
        r.font.bold = True
        r.font.size = Pt(header_font)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shd = hdr[i]._tc.get_or_add_tcPr()
        from docx.oxml import OxmlElement
        el = OxmlElement("w:shd")
        el.set(qn("w:val"), "clear"); el.set(qn("w:fill"), "1F4E79")
        shd.append(el)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(font)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    return t


# ----------------------------------------------------------------------------
# document
# ----------------------------------------------------------------------------

def build(out_path):
    doc = Document()
    style_doc(doc)
    add_page_number(doc)
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Cm(2.2)
    sec.left_margin = sec.right_margin = Cm(2.4)

    # ---- cover ----
    for _ in range(6):
        doc.add_paragraph()
    para(doc, "论 文 复 现 报 告", size=28, bold=True, color=ACCENT,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)
    para(doc, "《Discovering State-of-the-Art Quantum Circuits with Large Language Models》",
         size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    para(doc, "基于大语言模型的最先进量子电路发现 —— 完整独立复现",
         size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=24)

    info = [
        ("论文", "Andi Gu, Alek H. Kemeny, Susanne F. Yelin, Mikhail D. Lukin (Harvard / Anthropic)\narXiv:2610.22367 [quant-ph], 2026-10-15 (v1), 46 页, 10 图, 6 表"),
        ("官方代码", "https://github.com/andigu/uncompute (Apache-2.0),本地克隆于 ./uncompute"),
        ("首次完整复现", "2026-11-13(17/17 条目全部通过,与认证结果精确一致)"),
        ("实时复核", "2026-11-17(重新构建并评估 7 个电路,含 12.2× 头条结果,全部精确一致)"),
        ("评估器", "官方评估器 tools/eval.py;C 内核 (gcc-15 -O3 -march=native),执行器报告: kernel"),
        ("平台", "12 核 CPU,22 GiB RAM,Linux"),
        ("总体结论", "论文全部定量主张精确复现:17 个电路条目的 Toffoli 数、峰值量子比特数、TQ 成本与统计失败计数均与论文 Table I 及官方认证结果完全一致"),
    ]
    t = doc.add_table(rows=0, cols=2)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for k, v in info:
        cells = t.add_row().cells
        cells[0].text = ""
        r = cells[0].paragraphs[0].add_run(k)
        r.font.bold = True; r.font.size = Pt(10)
        cells[1].text = ""
        r = cells[1].paragraphs[0].add_run(v)
        r.font.size = Pt(10)
        cells[0].width = Cm(3.4); cells[1].width = Cm(12.4)
    doc.add_page_break()

    # ---- 1 executive summary ----
    doc.add_heading("1  执行摘要", level=1)
    para(doc, "本报告记录对 arXiv:2610.22367《Discovering state-of-the-art quantum circuits with large "
              "language models》的完整、独立复现。该论文提出一个演化搜索框架:由大语言模型(LLM)提出量子电路"
              "设计方案(以 Python 电路生成程序的形式),再由一个独立评估器在 kickmix 电路模型中验证并打分,"
              "最终以 TQ 成本(执行 Toffoli 数 T × 峰值活跃量子比特数 Q)对候选电路排序。")
    bullet(doc, "复现范围:官方仓库全部 17 个电路条目 —— Table I 的 8 个 LLM 发现电路、全部已收录基线、"
                "divstep Toffoli-量子比特权衡家族(j_store = 5…9、fill、wide)以及 SI II.2 的第二个欧几里得模逆电路。")
    bullet(doc, "复现方式:从源码 (circuit.py) 重新构建每个电路,用论文自带的评估器端到端再验证"
                "(随机化测试抽样下的正确性 + 精确成本计量),并逐项核对论文正文、官方认证结果 results.jsonl 与本次运行三方数值。")
    bullet(doc, "复现结果:所有运行均以 verdict OK 结束;每个执行 Toffoli 数、峰值量子比特数、TQ 成本与统计"
                "失败计数都与认证值及论文 Table I 精确一致(17/17 条目逐位匹配)。")
    bullet(doc, "实时复核(2026-11-17):在撰写本报告当天,重新构建并评估了 7 个电路(3 个数值任务的发现电路+基线,"
                "以及计算量最大的 dqi-reed-solomon 发现电路),全部精确复现,包括 442,397 × 1,837 的头条结果与"
                "100,000 次测试中恰好 7 次错误符号的统计行为。")
    bullet(doc, "论文核心主张全部得到确认:Reed-Solomon DQI 解码器 12.2×、HHL 旋转角 31.9×、库仑势 9.9×、"
                "定点高斯 39.1×、Fe2S2 化学别名-Select 4.1×、非阿贝尔格点规范 7.7× 的 TQ 成本缩减;"
                "首个可逆素域 divstep 模逆电路(406 万 Toffoli × 935 量子比特)及其可调 Toffoli-量子比特前沿;"
                "相对 luo-eea 基线 3.4× Toffoli 缩减的专用欧几里得变体;以及 Hermitian 解码器相对朴素可逆"
                "Feng-Rao 基线 9,114× 的改进(与 SI 声称的 9,100× 一致)。", )

    # ---- 2 paper & method ----
    doc.add_heading("2  论文与其方法", level=1)
    doc.add_heading("2.1  研究问题", level=2)
    para(doc, "量子算法的资源估计传统上依赖人类专家手工设计算术子电路(加法器、乘法器、查表、解码器等)。"
              "论文提出用 LLM 驱动的演化搜索自动发现比已知文献构造更优的算术/函数子电路,并以一个独立、"
              "可信的评估器作为唯一『正确性裁判』,从而避免 LLM 幻觉影响结论。")
    doc.add_heading("2.2  kickmix 电路模型", level=2)
    para(doc, "候选电路在 kickmix 模型中评估,该模型混合三类构件:")
    bullet(doc, "经典可逆门:X、CNOT、SWAP、Toffoli(CCNOT)、Fredkin 等可逆布尔逻辑,其中 Toffoli 计入成本,Clifford 免费计入统计;")
    bullet(doc, "基于测量的反计算(measurement-based uncomputation, MBU):对脏辅助比特做 X 基测量 (hmr) 并用测量结果"
                "条件化的相位门 (cz … cond=m) 修正符号,从而以零 Toffoli 代价清除辅助比特 —— 这是论文实现 4–39× 成本缩减的关键机制之一;")
    bullet(doc, "对角相位门:z / cz / ccz / neg 及按测量结果条件化的全局相位修正。")
    doc.add_heading("2.3  TQ 成本度量", level=2)
    para(doc, "电路成本定义为 TQ-cost = T × Q,其中 T 为执行的 Toffoli 门数(按门精确计量,非估计),"
              "Q 为峰值活跃量子比特数(由 LIFO 空闲表精确追踪)。该度量同时惩罚时间(门数)与空间(比特数),"
              "对应容错量子计算中 distillation 开销与数据量子比特两类主要资源。")
    doc.add_heading("2.4  演化搜索框架", level=2)
    para(doc, "搜索以 OpenEvolve 框架运行,Claude Mythos 5 作为提出模型,流程为:")
    bullet(doc, "构思 (ideation):LLM 仅凭任务规范、电路模型与正确性判据提出多个不同架构,每个架构形成一个独立的候选族;")
    bullet(doc, "迭代改写 (rewrite):每次迭代开启新的 LLM 会话,输入任务规范、父程序及其评估反馈、归档中的备选程序与"
                "历史洞见笔记,LLM 改写后提交;")
    bullet(doc, "评估:独立评估器构建电路、按 op 流的 SHA256 确定性抽取测试输入、在 kickmix 模拟器中执行全部测试,并输出"
                "分类诊断与精确 T、Q、TQ;")
    bullet(doc, "选择:MAP-Elites 将 (T, Q) 平面划分为网格,每格保留 TQ 最低的电路;族间定期共享领先程序,允许不同思路的组合。")
    doc.add_heading("2.5  论文的八个任务", level=2)
    tasks8 = [
        ("dqi-reed-solomon", "解码量子干涉 (DQI) 的 Reed-Solomon 解码器", "Berlekamp–Massey + 条件路由 + 解码历史复用"),
        ("dqi-hermitian", "DQI 的 Hermitian 码解码器", "Lanczos 型求解"),
        ("hhl-angle", "HHL 算法特征值依赖的旋转角 arcsin(C/x)", "dyadic 叶片 + 共享局部坐标 + 直接累加"),
        ("coulomb-potential", "库仑势 1/r 数值函数", "线段查表 (line table)"),
        ("gaussian", "定点高斯函数", "剪枝 QROM"),
        ("chemistry-fe2s2", "Fe2S2 铁硫簇电子结构的别名-Select", "frame-block 表示"),
        ("lattice-gauge-plaquette", "非阿贝尔格点规范理论 plaquette 项", "专用表示"),
        ("modinv-point-add", "素域模逆 + 椭圆曲线点加", "可逆 divsteps(首个可逆素域实现)"),
    ]
    make_table(doc, ["任务", "描述", "发现电路的核心思路"], tasks8,
               widths=[4.2, 6.4, 5.4])

    # ---- 3 scope & method ----
    doc.add_heading("3  复现范围与评估方法", level=1)
    doc.add_heading("3.1  评估流水线", level=2)
    para(doc, "每个电路条目均使用官方工具评估:")
    para(doc, "python tools/eval.py <task> [--circuit <path>]", size=10, color=ACCENT)
    para(doc, "该工具以子进程运行电路的 Python 构建器,将发出的 op 流经 FIFO 流式传输并同时做 SHA256 哈希"
              "(哈希用作测试抽样的种子),随后在 kickmix 模拟器中执行每个测试。当任务测试量超过批次上限时,"
              "评估器自动分片执行并在合并后给出与不分片完全一致的报告(分片/合并不变性)。")
    doc.add_heading("3.2  验收标准", level=2)
    bullet(doc, "输出寄存器在每个测试上与任务参考函数一致(错误输出计数在预算内,如 DQI 解码器允许 100/100,000,modinv 点加允许 20/100,000);")
    bullet(doc, "全局符号等于期望符号(错误符号计数同样受限,例如 Reed-Solomon 发现电路的 BM 解码器按设计允许 ~10⁻⁴ 的单条差异栈失败率);")
    bullet(doc, "所有非寄存器辅助量子比特返回 |0⟩(无脏比特);")
    bullet(doc, "成本(T、Q、TQ)从发出的 op 流精确计量,而非估算。")
    doc.add_heading("3.3  确定性与可复现性", level=2)
    para(doc, "评估器从发出 op 流的 SHA256 确定性导出测试输入,因此字节相同的重建会得到完全相同的测试套件。"
              "本次复现中所有重建均产生字节相同的 op 流(通过比对本次运行的 T/Q/TQ 与认证行验证;"
              "抽查电路的 h_ops 哈希亦一致)。")

    # ---- 4 environment ----
    doc.add_heading("4  复现环境", level=1)
    env = [
        ("Python", "3.12.16(uv venv);系统 Python 3.14.4"),
        ("依赖", "numpy、galois、mpmath、click 等(按官方 pyproject.toml)"),
        ("评估器内核", "官方 C 内核 uncompute/evalcore/_kernel.c,gcc-15 -O3 -march=native 编译;"
                    "由纯 Python 参考加载器加载。UNCOMPUTE_KERNEL 执行器报告:kernel"),
        ("主机", "12 核,22 GiB RAM,Linux"),
        ("并行性", "所有结果均不依赖并行度(工具声明分片/合并不变;C 内核仅为速度优化,纯 Python 参考评估器为规范实现)"),
    ]
    make_table(doc, ["项目", "配置"], env, widths=[3.6, 12.2])

    # ---- 5 results ----
    doc.add_heading("5  完整复现结果", level=1)
    doc.add_heading("5.1  Table I:发现电路与基线(17 个条目)", level=2)
    para(doc, "下表为论文 Table I 的全部条目。『认证』指官方仓库随附的 results.jsonl(由作者以 tools/certify.py 生成);"
              "『本次复现』指为撰写本报告所做的全新构建 + 端到端评估。每行均以 verdict OK 通过。")
    make_table(doc,
               ["任务", "电路", "论文 (T, Q)", "认证 (T × Q)", "本次复现 (T × Q)"],
               [(a, b, c, d, e) for (a, b, c, d, e) in TABLE1],
               widths=[3.4, 4.6, 2.7, 2.7, 2.7], font=8.5)
    para(doc, "17/17 条目与论文及认证值精确一致。", bold=True, color=GREEN)

    doc.add_heading("5.2  TQ 成本增益(按本次运行精确计数重算)", level=2)
    make_table(doc,
               ["任务", "基线 TQ", "发现电路 TQ", "论文 G_TQ", "本次 G_TQ"],
               GAINS, widths=[4.6, 3.6, 3.2, 2.2, 2.2])
    para(doc, "Hermitian 与 modinv 无单一基线增益(论文另行报告):Hermitian 基线 TQ 比为"
              " 20,232,614,305,126 / 2,219,954,240 = 9,114 倍,与 SI 的 9,100 倍主张一致;"
              "modinv 的欧几里得变体相对 luo-eea 基线实现 3.4 倍 Toffoli 缩减(69,529,757 → 20,521,949)。")

    doc.add_heading("5.3  divstep Toffoli-量子比特权衡家族(SI II.1,图 3 / 表 S3 前沿)", level=2)
    para(doc, "divstep 模逆通过 j_store 参数(存入栈的迭代变量数)在 Toffoli 数与量子比特数之间权衡;"
              "家族展示了可调的计算-空间前沿:")
    make_table(doc,
               ["条目", "T", "Q", "TQ", "本次复现"],
               [(a, b, c, d, "OK,精确一致") for (a, b, c, d) in DIVSTEP],
               widths=[4.4, 2.8, 1.8, 3.2, 3.6])
    para(doc, "家族中 TQ 最低点为 j_store=5 的 circuit.py(3,746,239,255),与论文选择的发布配置一致;"
              "j_store 增大单调降低 Q(935 → 893)但抬高 T,复现曲线与论文图 3 前沿形状一致。")

    doc.add_heading("5.4  统计失败行为(与论文报告的失败计数一致)", level=2)
    make_table(doc, ["电路 / 指标", "论文", "本次复现"], FAILURES, widths=[8.2, 2.2, 2.2])
    para(doc, "观察到的失败计数落在各评分器预算之内,并精确复现论文的陈述值 —— 包括有界误差设计"
              "(如 BM 解码器单条差异栈以 ~10⁻⁴ 概率失败)。")

    doc.add_heading("5.5  实时复核(2026-11-17,撰写本报告当日)", level=2)
    para(doc, "为排除『仅转抄历史结果』的可能,撰写本报告时重新从源码构建并完整评估了下列电路"
              "(全部使用官方评估器与 C 内核):")
    make_table(doc,
               ["电路", "verdict", "T", "Q", "TQ", "与认证值比对"],
               LIVE, widths=[5.0, 1.6, 1.6, 1.2, 2.4, 4.0], font=8.5)
    para(doc, "其中 dqi-reed-solomon 发现电路的实时复核耗时约 5 分 36 秒(12 核,C 内核),"
              "100,000 次测试中恰好出现 7 次错误符号 —— 与论文及认证值逐位一致。")

    # ---- 6 notes ----
    doc.add_heading("6  复现技术说明", level=1)
    bullet(doc, "字节级确定性:测试输入由 op 流的 SHA256 播种,字节相同的重建产生相同测试套件;"
                "所有重建的 op 流与认证行一致(h_ops 哈希抽查一致)。")
    bullet(doc, "C 内核仅为速度优化:纯 Python 参考评估器是规范实现(官方 README);本次复现使用 C 内核。")
    bullet(doc, "chemistry-fe2s2 运行 18,419,296 次测试(1,642,080 次结构化探测 + 2²⁴ 次随机);"
                "其发现电路与基线均以零错误输出/零错误符号通过。")
    bullet(doc, "dqi-hermitian 的 sakata-fengrao 基线是评估过的最大电路:104,263,342 个 Toffoli、"
                "194,053 个量子比特、TQ ≈ 2.0×10¹³;使用 C 内核在约 21 分钟内通过 100,000 次测试,0 失败。")
    bullet(doc, "分片/合并不变性使得本复现的全部数值与并行度无关。")

    # ---- 6b optimality certificate (new 2026-11-17) ----
    doc.add_heading("6b  论文 F24 演示电路的最优性证书(新,2026-11-17)", level=1)
    para(doc, "在精确复现之外,本 harness 补充了论文所没有的内容:对论文自身演示任务"
              "(F24 求逆,扩展数据图 1;论文发现电路 T=5、Q=10、TQ=50)的机器可校验最优性证书。"
              "证书模块:sota_harness/sota_harness/certificates.py;全部数值可由 run_all.sh 复现。")
    make_table(doc,
               ["证书支腿", "陈述", "模型", "证据"],
               [
                   ("T ≥ 5(任意 Q)",
                    "任何量子比特数下都不存在 4-Toffoli 电路",
                    "kickmix 模型;辅助比特数不受限;允许 h 位作为操作数来源;零相位",
                    "秩证明:rank span{d2(INV_b)} = 4;完整的 171 个首操作乘积菜单中"
                    "31 个 d2=0、140 个在目标空间之外、0 个非零在内;4 乘积张成 4 维"
                    "目标空间必须全部落在其中 —— 矛盾。2 辅助比特穷举 DFS 交叉验证"
                    "(tsearch,343 节点,0.01 秒)"),
                   ("Q ≥ 9(任意 T,严格模型)",
                    "仅以受瞬态整形的 x 线为操作数时,任意 Toffoli 数下均无 0 辅助比特电路",
                    "操作数 = 受整形 x 线;h 位仅作目标",
                    "四个目标函数的 d2 向量位于 35 个可实现 x 对乘积 d2 所张成的 6 维"
                    "子空间之外(全空间 11 维);q8probe.py"),
                   ("Q ≥ 10(T=5)",
                    "论文操作数习语下不存在 1 辅助比特、5 Toffoli 电路",
                    "单辅助比特 t;t 操作用 x 仿射操作数对(MBU 可释放);h 操作用 x 对或"
                    "(t 前缀 & x 仿射)对;自由快照与仿射装配",
                    "穷举确定性 DFS:NOT FOUND,恰好 6,725,331 节点(两次运行完全一致);"
                    "q9search.py 5"),
                   ("存在性", "T=5、Q=10 电路存在", "—",
                    "论文电路(扩展数据图 1c),(T, Q) 如论文发表值"),
               ],
               widths=[2.4, 3.4, 4.2, 6.2], font=8)
    para(doc, "加强与拓广:T ≥ 5 支腿由『2 辅助比特』升级为『任意 Q』(秩论证),且覆盖 h 位作为操作数来源。"
              "塔域/Frobenius 攻击:自由平方/基变换不扩大乘积菜单(仿射形式的乘积在复合下封闭),"
              "x⁻¹ = x¹⁴ 的最优塔链仍需 2 次 GF(16) 乘法 = 8 个 Toffoli > 5,为否定结果。"
              "放宽模型搜索(Q=8 与 Q=9 允许 h 位作源;q8hsrc.py、q9hsrc.py)均未在"
              "计算预算内闭合,如实记录为 TIMEOUT/无结论(并非 NOT-FOUND 证明):"
              "Q=8 达 13,300,000 节点(7,225 秒);Q=9 达 24,200,000 节点(14,407 秒,"
              "4 小时截止)。两个已探索空间中均未发现更优电路 —— 与证书一致但非其证明。"
              "最终定稿的健全双辅助比特模型(existsearch.py v3:辅助比特目标操作仅限 MBU 安全的"
              "x 仿射对,h 目标操作含 x/t/u/tu/h 源族,t/u 真值表状态,含基的记忆化)穷举闭合:"
              "NOT FOUND,74,594,457 节点(1,644 秒,18,225 个记忆状态)—— 这是 Q=10、T=5 在"
              "所搜索最宽健全子类中的强否定结论(严格强于上述两个 TIMEOUT)。需明确限定:该模型是"
              " Q=10 电路的真子类,论文自身的 (T=5, Q=10) 电路必然使用了该子类之外的操作,故该"
              "穷举闭合只断言『此模型内无电路』,而非『不存在两辅助比特电路』;模型中无电路,"
              "故无候选通过『发射 + f24_evaluator.py 裁决』门槛;存在性支腿仍采用论文发表电路。"
              "物理搜索(physsearch.py)亦在 13.3M+ 节点后无结论(无截止时间,人为终止;"
              "存在性支腿采用论文发表电路)。")
    para(doc, "记录更正:两个搜索脚本曾携带早期会话的过时乐观 docstring(finalsearch.py 的"
              "『TQ = 45 < 50!』、tsrcsearch.py 的『better than the paper's 50』),"
              "而其实际运行结果均为 NOT FOUND;已修正为如实陈述否定结果。"
              "candidates/ 目录除新增 F22/F23 塔演示外为空;仓库中不再存在未经验证的电路断言。")

    doc.add_heading("6c  证书塔:GF(4) 与 GF(8) 求逆(新,2026-11-17)", level=1)
    para(doc, "同一套机器再对论文 F24 任务族向下两个规模出具完整证书(全部经官方 Builder/Simulator "
              "在全部 (x, h) 对上穷举验证;verify_f22_f23.py):")
    make_table(doc,
               ["任务", "构造", "(T, Q)", "TQ", "T 最优性", "发现方式"],
               [
                   ("F22 求逆 (GF(4))", "Frobenius:z⁻¹ = z²,纯 Clifford", "(0, 4)", "0",
                    "秩 0,平凡最优", "确定性(代数)"),
                   ("F23 求逆 (GF(8))", "逐位张成分解:x1&x2、x0&x1、x0&x2 + 仿射残差", "(3, 6)", "18",
                    "秩证明:rank span{d2(INV_b)} = 3 ⇒ 任意 Q 下无 2-Toffoli 电路", "确定性(逐位 d2 匹配),无 LLM"),
                   ("F24 求逆 (GF(16))", "论文 LLM 发现电路", "(5, 10)", "50",
                    "秩 4 + 菜单排除 ⇒ 任意 Q 下无 4-Toffoli 电路", "论文 LLM 演化"),
               ],
               widths=[3.0, 4.6, 1.6, 1.2, 4.4, 2.0], font=8)
    para(doc, "Table-I 任务(hhl-angle、coulomb 等)的完整证书对穷举方法不可达(输入宽度 21–48 位),"
              "此处如实陈述而不作声称。", size=10)

    # ---- 7 conclusion ----
    doc.add_heading("7  结论", level=1)
    para(doc, "官方 andigu/uncompute 仓库的全部 17 个电路条目 —— 论文 Table I 的八个 LLM 发现电路、"
              "全部收录基线、以及完整的 divstep Toffoli-量子比特前沿 —— 均已从源码重建并以论文自带的精确评估器"
              "端到端再验证。每个执行 Toffoli 数、峰值量子比特数、TQ 成本与统计失败计数均与认证值及论文 Table I "
              "精确一致。复现确认了论文的全部头条主张:")
    for s in (
        "Reed-Solomon DQI 解码器 12× TQ 成本缩减(Berlekamp–Massey + 条件路由 + 解码历史);",
        "HHL 旋转角 32×(dyadic 叶片 + 共享局部坐标 + 直接累加);",
        "库仑势 9.9×、定点高斯 39.1×、Fe2S2 化学 4.1×、格点规范 7.7×;",
        "首个可逆素域 divstep 模逆(406 万 Toffoli × 935 量子比特),附可调 Toffoli-量子比特前沿,"
        "以及 3.4× Toffoli 的专用欧几里得变体;",
        "Hermitian 解码器相对朴素可逆 Feng-Rao 基线 9,100× 的改进。",
    ):
        bullet(doc, s)
    para(doc, "在确认复现的基础上,后续工作(sota_harness)将在论文框架之上构建更强的发现系统:"
              "确定性代数提议器(ANF/张成空间综合)、Toffoli 下界证书(乘法复杂度)、Pareto 前沿归档、"
              "分类失败诊断与可复现 JSONL 日志等超出论文范围的能力。", size=10)

    # ---- appendix ----
    doc.add_heading("附录 A  复现命令", level=1)
    cmds = [
        ("评估任意任务(默认发现电路)", "python tools/eval.py <task> [--circuit PATH] --json"),
        ("评估基线", "python tools/eval.py hhl-angle --circuit circuits/haner-piecewise/circuit.py"),
        ("认证全部条目(官方)", "python tools/certify.py(生成 results.jsonl)"),
        ("指定执行器", "UNCOMPUTE_KERNEL=1 强制 C 内核;=0 强制纯 Python 参考实现"),
    ]
    make_table(doc, ["操作", "命令"], cmds, widths=[5.0, 10.8], font=9)

    doc.save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/david/software/sota_quantum_circuits/复现报告_Replication_Report.docx"
    build(out)

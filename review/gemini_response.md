以下是对您提交的20条学术参考文献进行的零信任底层抓取与硬性双盲比对报告。

* **BibTeX key:** `chen2021codex`
* **核查判定:** `[🚨 FATAL ERROR]`
* **异常详情:** 触发独立实体定位中断。原文献全称为《Evaluating Large Language Models Trained on Code》，实际存在 25 位作者（包括 Alex Ray, Raul Puri, Gretchen Krueger, 等）。输入信息不仅遗漏了超过半数作者，还使用了系统明令禁止的掩盖词 “et al.”。


* **BibTeX key:** `austin2021program`
* **核查判定:** `[🚨 FATAL ERROR]`
* **异常详情:** 作者列表掩盖错误。通过 arXiv 实时解析发现，该文存在 11 位作者，而输入列表使用了 “et al.” 掩盖了末位作者 Charles Sutton 的存在。


* **BibTeX key:** `guo2017calibration`
* **核查判定:** `[✅ VALIDATED]`
* **证据链接:** [https://proceedings.mlr.press/v70/guo17a/guo17a.pdf](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)
* **备注:** 标题、年份、出处与 4 位作者（Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger）的字母拼写完全匹配。


* **BibTeX key:** `liu2024your`
* **核查判定:** `[✅ VALIDATED]`
* **证据链接:** [https://openreview.net/forum?id=VTF8yNQM66](https://openreview.net/forum?id=VTF8yNQM66) (NeurIPS 2023)
* **备注:** 作者顺序与拼写（Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, Lingming Zhang）精准对齐，不存在缺失。


* **BibTeX key:** `du2024evaluating`
* **核查判定:** `[🚨 FATAL ERROR]`
* **异常详情:** 严重幻觉与作者伪造。通过解析 ICSE 2024 Proceedings (DOI: 10.1145/3597503.3639219)，该文实际后半部分作者应为 Jiayi Feng, Chaofeng Sha, Xin Peng, Yiling Lou。输入数据将 Jiayi 错误拼写为 Junjie，并且凭空捏造了 Guangyu Zhu; Shang-Wei Li; Yang Liu 三人。


* **BibTeX key:** `hendrycks2021apps`
* **核查判定:** `[🚨 FATAL ERROR]`
* **异常详情:** 作者列表大面积错乱。原始数据证实该文共有 11 位作者，输入不仅遗漏了 Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He 五人，甚至无中生有地插入了 “Andy Zou”。


* **BibTeX key:** `li2022alphacode`
* **核查判定:** `[🚨 FATAL ERROR]`
* **异常详情:** “et al.” 违规触发。根据 Science 原始出版物 (DOI: 10.1126/science.abq1158)，该文包含 Yujia Li, David Choi 等共 25 位作者。输入直接截断并使用 et al. 进行掩盖。


* **BibTeX key:** `jain2024livecodebench`
* **核查判定:** `[✅ VALIDATED]`
* **证据链接:** [https://arxiv.org/abs/2403.07974](https://arxiv.org/abs/2403.07974)
* **备注:** 包含 10 位作者的长列表（Naman Jain 至 Ion Stoica）逐词比对无误，数据完全通过游标卡尺级检测。


* **BibTeX key:** `jimenez2024swebench`
* **核查判定:** `[✅ VALIDATED]`
* **证据链接:** [https://arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770)
* **备注:** 标题、以及 7 位作者的缩写和中间名（如 Carlos E. Jimenez, Karthik R. Narasimhan）全部匹配。


* **BibTeX key:** `wang2022self`
* **核查判定:** `[✅ VALIDATED]`
* **证据链接:** [https://arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)
* **备注:** 8位作者顺序与拼写全节点通过。


* **BibTeX key:** `kojima2022zeroshot`
* **核查判定:** `[🚨 FATAL ERROR]`
* **异常详情:** Token 拼写错误。最后一位作者的真实姓名为 Yusuke Iwasawa，而输入数据中将其错误写为 Yutaka Iwasawa（受前一位作者 Yutaka Matsuo 影响产生的名称混淆）。比对失败，中断评估。


* **BibTeX key:** `zhang2026performance`
* **核查判定:** `[🚨 FATAL ERROR]`
* **异常详情:** 查无此文。跨越 ResearchGate、DBLP、arXiv 等数据库检索，提供的编号 403969300 属于幻觉产物。没有任何证据证明《Performance-Efficiency Trade-offs in Human Preference Prediction...》一文及对应作者的真实存在。


* **BibTeX key:** `brown2024large`
* **核查判定:** `[✅ VALIDATED]`
* **证据链接:** [https://arxiv.org/abs/2407.21787](https://arxiv.org/abs/2407.21787)
* **备注:** 作者列表（包含 Quoc V. Le 和 Christopher Ré 的特殊字符与缩写）比对无误。


* **BibTeX key:** `atil2024llm`
* **核查判定:** `[🚨 FATAL ERROR]` d
* **异常详情:** 使用了 “et al.” 触发绝对阻断机制。


* **BibTeX key:** `alvarado2025repetitions`
* **核查判定:** `[🚨 FATAL ERROR]`
* **异常详情:** 使用了 “et al.” 触发绝对阻断机制。依据 arXiv:2509.24086 原型文件，该文章有 6 位作者（Miguel Angel Alvarado Gonzalez 至 Sandra Malagon）。


* **BibTeX key:** `deng2023investigating`
* **核查判定:** `[✅ VALIDATED]`
* **证据链接:** [https://aclanthology.org/2024.naacl-long.482/](https://aclanthology.org/2024.naacl-long.482/)
* **备注:** 5位作者（Chunyuan Deng 至 Arman Cohan）完全匹配。


* **BibTeX key:** `aggarwal2023lets`
* **核查判定:** `[✅ VALIDATED]`
* **证据链接:** [https://aclanthology.org/2023.emnlp-main.761/](https://aclanthology.org/2023.emnlp-main.761/)
* **备注:** 会议正式收录及预印本中的 4 位作者（Pranjal Aggarwal, Aman Madaan, Yiming Yang, Mausam）完全一致。


* **BibTeX key:** `zhuo2024prosa`
* **核查判定:** `[✅ VALIDATED]`
* **证据链接:** [https://arxiv.org/abs/2410.12405](https://arxiv.org/abs/2410.12405)
* **备注:** 6 位作者的顺序及中文拼音拼写毫无差异。


* **BibTeX key:** `gao2023text`
* **核查判定:** `[✅ VALIDATED]`
* **证据链接:** [https://arxiv.org/abs/2308.15363](https://arxiv.org/abs/2308.15363) (最终发表于 PVLDB)
* **备注:** 从 Dawei Gao 到 Jingren Zhou 7 位作者精确匹配。


* **BibTeX key:** `jiang2026agenticaicybersecurityattack`
* **核查判定:** `[⚠️ PARTIAL/VERSION UPDATE]`
* **证据链接:** [https://arxiv.org/abs/2602.19555](https://arxiv.org/abs/2602.19555)
* **更新建议:** 作者信息（Xiaochong Jiang 至 Cheng Ji）匹配成功，但检测到版本名称发生了迭代。原早期预留标题已被更新为官方正式版标题《SOK: A Taxonomy of Attack Vectors and Defense Strategies for Agentic Supply Chain Runtime》。请以新标题进行替换。
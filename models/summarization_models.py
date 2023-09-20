"""Model classes for Summarization Task
BART LARGE CNN:
    text = ''' New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
    A year later, she got married again in Westchester County, but to a different man and without divorcing her first
    husband. Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more
    times, sometimes only within two weeks of each other.
    In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it
    was her "first and only" marriage.
    Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree,"
    referring to her false statements on the 2010 marriage license application, according to court documents.
    Prosecutors said the marriages were part of an immigration scam.
    On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher
    Wright, who declined to comment further. After leaving court, Barrientos was arrested and charged with theft of
    service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said
    Detective Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her
    marriages occurring between 1999 and 2002. All occurred either in Westchester County, Long Island, New Jersey or
    the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once,
    prosecutors say. Prosecutors said the immigration scam involved some of her husbands, who filed for permanent
    residence status shortly after the marriages. Any divorces happened only after such filings were approved. It was
    unclear whether any of the men will be prosecuted. The case was referred to the Bronx District Attorney\'s Office
    by Immigration and Customs Enforcement and the Department of Homeland Security\'s Investigation Division. Seven of
    the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali. Her eighth
    husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism
    Task Force. If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for
    May 18.
    '''
    summarizer = SummarizationModel("facebook/bart-large-cnn")
    summarizer.predict(text, max_length=130, min_length=30, do_sample=False)

MT5 Multilingual:
    text = '''救出作戦の間、洞窟内に少年たちと留まったタイ海軍のダイバーと医師も最後に無事脱出した。4人の写真は10日、
    タイ海軍特殊部隊がフェイスブックに掲載したもの タイ海軍特殊部隊はフェイスブックで、「これは奇跡なのか科学なのか、
    一体何なのかよくわからない。『イノシシ』13人は全員、洞窟から出た」と救助作戦の終了を報告した。「イノシシ」（タイ語で「ムーパ」）
    は少年たちの所属するサッカー・チームの愛称。 遠足に出かけた11歳から17歳の少年たちと25歳のサッカー・コーチは6月23日、
    大雨で増水した洞窟から出られなくなった。タイ内外から集まったダイバー約90人などが捜索に当たり、英国人ダイバー
    2人によって7月2日夜に発見された。地元のチェンライ県知事やタイ海軍特殊部隊が中心となった救助本部は当初、水が引くか、
    あるいは少年たちが潜水技術を習得するまで時間をかけて脱出させるつもりだったが、雨季による水位上昇と洞窟内の酸素低下の進行が懸念され
    、8日から3日連続の救出作戦が敢行された。 少年たちの脱出方法 ダイバーたちに前後を支えられ、水路内に張り巡らされたガイドロープをたどりながら
    、潜水経験のない少年たちは脱出した。8日に最初の4人、9日に4人、10日に残る5人が脱出し、ただちに近くのチェンライ市内の病院に搬送された。
    2週間以上洞窟に閉じ込められていたことを思えば、全員驚くほど心身ともに元気だという。 少年たちとコーチはレントゲンや血液検査などを受けた。
    少なくとも7日間は、経過観察のために入院を続けるという。 洞窟内の水を飲み、鳥やコウモリの排泄物に接触した可能性のある13人は、
    病原体に感染しているおそれがあるため隔離されている。家族とはガラス越しに再会したという。
    食べ物のほとんどない洞窟内で2週間以上を過ごした少年たちは体重を大幅に落とし、空腹を訴えていた。救出後は好物の豚肉のご飯やパン、
    チョコレートなどを希望したが、しばらくは流動食が続くという。 さらに、外界の光に目が慣れるまでの数日は、サングラスをかける必要がある。
    ＜おすすめ記事＞ 救出作戦が終わると、洞窟の出口に集まった救助関係者から大きな歓声が上がった。山のふもとには、少年たちが所属する
    「ムーパ（イノシシ）」サッカーチームの関係者の家があり、そこに集まった人たちも笑顔で叫んだり歓声を挙げたりした。現場にいたBBCのジョナサン・
    ヘッド記者は、喜ぶ人たちは「とてもタイ人らしくない様子で」さかんに握手をして回っていたと伝えた。 少年たちへの精神的影響は？ タイ洞窟救助
    チェンライ市では、全員脱出の知らせに往来の車は次々にクラクションを鳴らして喜んだ。子供たちやコーチが搬送された病院の外に集まっていた人たち
    は、一斉に拍手した。 ソーシャルメディアではタイ人の多くが、「#Heroes(英雄）」、「 #Thankyou（ありがとう）」などのハッシュタグを使って、
    それぞれに思いを表現していた。 13人は2日、洞窟内の岩場に身を寄せているところを発見された。中央の少年は、サッカーのイングランド代表のシャ
    ツを着ている。写真はタイ海軍が4日に公表したビデオより サッカー界も少年たちとコーチの無事を大いに喜び、英マンチェスター・ユナイテッドやポル
    トガルのベンフィカが全員を試合に招待した。国際サッカー連盟（FIFA）も、少年たちをロシアで開催されているワールドカップの15日にある決勝戦に
    招いたが、これは回復が間に合わないという理由で見送られた。 ワールドカップの準決勝に備えるイングランド代表のDFカイル・ウォーカーは、イング
    ランドのユニフォームを少年たちに贈りたいとツイートした。少年の1人は洞窟内で、イングランドのジャージーを着ていた。すると英外務省の公式アカ
    ウントがこれに応えて、「やあ、カイル。駐タイ英国大使と話をした。イングランドのシャツを勇敢な少年たちに、喜んで、確実に届けてくれるそうだ」
    とツイートした。 経験豊富なダイバーにとっても、少年たちのいる場所までの往復は重労働だった。元タイ海軍潜水士のサマン・グナンさんは6日、少年
    たちに空気ボンベを運ぶ任務を果たして戻ろうとしていたところ、酸素不足で命を落とした。 ダイバーたちが出口まで張ったガイドロープをたどりなが
    ら、少年たちは場所によって、歩いたり、水の中を歩いたり、登ったり潜ったりして外に出た。 少年たちは、通常のマスクよりも初心者に適した顔部全
    体を覆うマスクをかぶった。少年1人につき2人のダイバーが付き、ダイバーが少年の空気ボンベを運んだ。 最も困難なのは、洞窟の中ほどにある「Tジ
    ャンクション」と呼ばれている場所で、あまりに狭いため、ダイバーは空気ボンベを外して進む必要があった。 Tジャンクションを抜けると、ダイバー
    達の基地となっている「第3室」があり、少年たちはここで出口へ向かう前に休息がとれた。 少年らの救出経路。下方の赤い丸が少年たちの見つかった
    場所。人の形が実際の人間の身長。青い部分は潜水しないと進めない。高さが1メートルに満たない箇所もある。トンネル内で最も狭い部分は、人1人が
    やっと通れるぐらいのスペースしかない。上方の白い部分は、ところどころ浅い水があるが、ほとんどが乾いた岩場 （英語記事 Cave rescue:
     Elation as Thai boys and coach freed by divers）
    '''
    summarizer = SummarizationModel("csebuetnlp/mT5_multilingual_XLSum")
    summarizer.predict(text)
"""
from base import HuggingFaceModel
from transformers import pipeline

TASK = "summarization"


class SummarizationModel(HuggingFaceModel):
    def __init__(self, model_name):
        HuggingFaceModel.__init__(self)
        self.model_name = model_name
        self.summarizer = pipeline(TASK, model=self.model_name)

    def predict(self, text, **kwargs):
        response = self.summarizer(text, **kwargs)
        return response[0]["summary_text"]

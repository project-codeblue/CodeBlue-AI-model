# 증상 데이터와 응급 정도 레이블
data = [
    ("밥을 먹다가 아랫입술이 경련이 난 것처럼 떨린다.", 4),
    ("10분 정도 한 후에 괜찮아짐.", 2),
    ("월요일 체력단련 후 명치가 아프면서 밤새 동안 구토, 구역질", 3),
    ("교정하고 있는 상태.", 1),
    ("운동을 하다가 심장이 갑자기 아프다.", 3),
    ("증상은 1주일 정도 계속", 3),
    ("가끔씩 어지럽거나 빈혈도 있음", 2),
    ("혈압약을 먹고 있음", 1),
    ("아스팔트에서 넘어져 살이 까지는 찰과상", 2),
    ("다음날 상처 부위에 진물이 흐름", 1),
    ("지금은 상처 주변이 빨갛게 변함", 2),
    ("상처에서 열감이 느껴짐", 2),
    ("담석증으로 인해 복강경 수술 후 3일 뒤 퇴원", 3),
    ("역류성 식도염 증상과 구역감이 지속", 3),
    ("식사 후 구역감이 계속 생김", 3),
    ("역류성 식도염처럼 목에 이물감이 계속 남아있음", 2),
    ("4일 전 부터 잇몸이 너무 아프고 시림", 2),
    ("출혈이 반복되며 소염 진통제를 먹어도 효과가 없음", 4),
    ("잇몸이 헐고 열이 계속 남", 2),
    ("바람이 불면 상처부위랑 얼굴이 아프고 식사를 제대로 못함", 3),
    ("자고 일어났을 때 목과 어깨가 뻐근하고 찌릿한 느낌을 받음", 1),
    ("오른쪽 귀 밑부터 쇄골 윗 부분 근육까지 아픔", 2),
    ("교통사고 난 느낌일 정도로 아픔", 3),
    ("얼굴 광대 아랫부분부터 입 주변 부분이 저림", 2),
    ("통증은 없으며, 지릿지릿한 느낌이고 증상이 한번 발생하면 보통 3~5시간 지속", 2),
    ("정신과에 불안장애 진단을 받고 현재 약을 복용중", 3),
    ("최근 들어 과호흡이 심해짐", 3),
    ("호흡곤란 느낌이 이어지며 심장이 답답하고, 약을 복용 후에도 지속", 3),
    ("일상생활이 힘든 정도", 3),
    ("회사 근무 간 불안과 긴장, 한숨이 반복", 1),
    ("본인의 직책에 대한 과도한 불안감", 2),
    ("본인의 적응 능력에 대한 자책과 울적함을 느끼며, 과한 음주", 2),
    ("긴장 완화를 위해 본인에게 맞지 않는 커피를 지속 섭취", 1),
    ("피를 뽑은 후 바늘 구멍 주변이 모기물린 것 처럼 부어오름", 1),
    ("간지럽고 1~2달간 상처가 아물지 않음", 2),
    ("이전에 극 민감성 아토피를 앓았던 적이 있음", 2),
    ("6개월이 지난 지금은 주사부위만 트고 가끔 가려움", 1),
    ("목을 오른쪽으로 꺾으면 가슴 명치 쪽이 쑤심", 2),
    ("수면 중 기침 때문에 자다가 숨 넘어갈 것 같음", 2),
    ("명치 왼쪽 부분이 쑤시는 느낌이 듬", 2),
    ("정수리 오른쪽 뾰루지 부분에 바늘로 찌르는 듯한 심한 통증을 느낌", 2),
    ("앞으로 휘청거려서 쓰러질 정도로 중심을 못잡음", 3),
    ("심장이 따끔 거리고 호흡이 힘듬", 3),
    ("의식이 없으며, 자가 호흡을 하지 못함", 5),
    ("팔에 심한 출혈이 있으며, 맥박이 떨어짐", 5),
    ("의식이 없으며 심박이 불안함", 5),
    ("환자는 경기를 일으키고 있음", 4),
    ("공황장애 병력을 갖고 있음", 3),
    ("체온이 낮고 나른함", 1),
    ("종아리가 심하게 부어있음", 2),
    ("숨을 쉴 때 갑자기 흉부에 통증을 느낌", 2),
    ("마른 기침이 나오며 숨을 쉬기 힘들어 합니다.", 3),
    ("평소에 사과 알레르기가 있으며, 눈과 입술이 부어오름", 2),
    ("원인 미상의 알레르기 반응으로 목 부분이 심하게 부었음", 3),
    ("동공이 확장되었으며 식은 땀을 흘림", 3),
    ("목이 심하게 부어 숨을 쉬는데 고통을 호소하고 있음", 3),
    ("맥박이 떨어지고 있으며 의식이 없음", 5),
    ("심한 두통과 고열 증세를 보임", 2),
    ("몸이 건조하며 탈수 증상이 있음", 2),
    ("정신착란 증세가 있으며 사물을 인지 못함", 4),
    ("강한 뇌진탕으로 인하여 의식이 없음", 5),
    ("엉덩이부근에 심한 찰과상이 있음", 2),
    ("소실된 의식과 기억 상실, 혼란상태를 보여요.", 4),
    ("환자가 오늘 아침에 발열이 높았다. 또한 환자가 혈압이 높았다.", 2),
    ("두통으로 인해 치매 증상 발생", 3),
    ("두통때문에 혼절함", 4),
    ("두통, 치통이 있으며 오른쪽 팔목에 통증을 호소한다.", 2),
    ("기억 상실상태이며, 가끔 가슴통증도 동반한다", 2),
    ("기운이 없으며 손발이 계속 저리고, 청각이 손실 된듯 하다", 3),
    ("피를 계속 토하고, 각혈이 지속되다보니 혈압이 높아져서 쓰러짐", 5),
    ("심부전의 징후인 가슴 통증, 사지 부종, 코막힘이 있어요.", 4),
    ("발작으로 인한 청각 손실, 시야 손실 발생했습니다.", 4),
    ("호흡곤란과 호흡음, 흉부 압박감을 겪고 있습니다.", 4),
    ("고열과 목의 부종 사지 마비로 도움이 필요해 보입니다.", 5),
    ("팔과 다리 부분에 1도 화상을 입었습니다.", 2),
    ("전신에 힘이 없으며 몸을 제대로 가누지를 못함", 2),
    ("혈압이 높고 경련 증세가 있음", 3),
    ("사물을 인지하지 못하고 상황판단이 불가능함", 3),
    ("엉덩이에서 작은 출혈이 있음", 1),
    ("등에 멍자국이 있음", 1),
    ("얼굴에 붉은 반점이 있음", 1),
    ("안면부에 심한 찰과상이 있으며 출혈이 심합니다", 3),
    ("기침에 피를 동반함", 3),
    ("어깨 부근의 탈골이 의심됨.", 1),
    ("빈혈끼가 있으며 시야가 보이지 않음", 3),
    ("앞가슴을 심하게 누르는 듯한 느낌을 받음", 3),
    ("환자의 전신에 자해 자국이 있음", 3),
    ("다소 심한 뇌진탕으로 인해 의식이 희미함", 4),
    ("목 부근에 자해의 흔적이 있다", 4),
    ("코피 음식 섭취 곤란, 가려운 발진이 관찰되어요.", 2),
    ("피토 하고, 각혈이 점점 심해져서 즉각적인 응급 처치가 필요함", 5),
    ("오전에 발작을 일으켰으며 기운이 없어보이고, 호흡곤란이 있엇다", 4),
    ("빈혈과 혈액 흘림, 황달 증상을 확인했습니다.", 3),
    ("경련, 목의 부종, 의식 변화로 급한 조치가 필요해보입니다.", 5),
    ("환자는 객혈과 피토를 겪고 있습니다.", 5),
    ("기침을 자주하며 목이 심하게 부어오름", 1),
    ("무릎에 심한 출혈이 있습니다", 3),
    ("전신에 격통을 느끼고 있음", 4),
    ("심한 발열과 함께 방금 있었던 일을 기억하지 못하는 등의 증세를 보임",5),
    ("자가 호흡이 불가능하며 빠른 이송이 필요",5),
]
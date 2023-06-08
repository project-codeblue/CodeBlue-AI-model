"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var child_process_1 = require("child_process");
const iconv = require("iconv-lite");

try {
    // Python 스크립트 실행하여 값을 받아오는 예시
    var result = (0, child_process_1.execSync)('python trans.py'); // 해당 python 파일명 입력
    var output = iconv.decode(result, 'euc-kr').trim(); // 해당 인코딩 값에 맞게 변경후 디코딩
    console.log(output); // 받아온 값을 출력
}
catch (error) {
    console.error(error);
}
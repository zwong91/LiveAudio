import { useState } from 'react';

interface LanguageSelectionProps {
  onLanguageChange: (sourceLang: string, targetLang: string) => void;
}

const LanguageSelection = ({ onLanguageChange }: LanguageSelectionProps) => {
  const [sourceLang, setSourceLang] = useState('en');
  const [targetLang, setTargetLang] = useState('en');

  const handleSourceLangChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newSourceLang = e.target.value;
    setSourceLang(newSourceLang);
    onLanguageChange(newSourceLang, targetLang); // 通知父组件更新语言
  };

  const handleTargetLangChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newTargetLang = e.target.value;
    setTargetLang(newTargetLang);
    onLanguageChange(sourceLang, newTargetLang); // 通知父组件更新语言
  };

  return (
    <div>
      <div>
        <label htmlFor="sourceLang">Source Language:</label>
        <select
          id="sourceLang"
          value={sourceLang}
          onChange={handleSourceLangChange}
        >
          <option value="en">English (en)</option>
          <option value="zh-cn">Chinese (zh-cn)</option>
          <option value="ko">Korean (ko)</option>
          <option value="ja">Japanese (ja)</option>
          <option value="es">Spanish (es)</option>
          <option value="fr">French (fr)</option>
          <option value="de">German (de)</option>
          <option value="it">Italian (it)</option>
          <option value="pt">Portuguese (pt)</option>
          <option value="pl">Polish (pl)</option>
          <option value="tr">Turkish (tr)</option>
          <option value="ru">Russian (ru)</option>
          <option value="nl">Dutch (nl)</option>
          <option value="cs">Czech (cs)</option>
          <option value="ar">Arabic (ar)</option>
          <option value="hu">Hungarian (hu)</option>
          <option value="hi">Hindi (hi)</option>
        </select>
      </div>

      <div>
        <label htmlFor="targetLang">Target Language:</label>
        <select
          id="targetLang"
          value={targetLang}
          onChange={handleTargetLangChange}
        >
          <option value="en">English (en)</option>
          <option value="zh-cn">Chinese (zh-cn)</option>
          <option value="ko">Korean (ko)</option>
          <option value="ja">Japanese (ja)</option>
          <option value="es">Spanish (es)</option>
          <option value="fr">French (fr)</option>
          <option value="de">German (de)</option>
          <option value="it">Italian (it)</option>
          <option value="pt">Portuguese (pt)</option>
          <option value="pl">Polish (pl)</option>
          <option value="tr">Turkish (tr)</option>
          <option value="ru">Russian (ru)</option>
          <option value="nl">Dutch (nl)</option>
          <option value="cs">Czech (cs)</option>
          <option value="ar">Arabic (ar)</option>
          <option value="hu">Hungarian (hu)</option>
          <option value="hi">Hindi (hi)</option>
        </select>
      </div>
    </div>
  );
};

export default LanguageSelection;

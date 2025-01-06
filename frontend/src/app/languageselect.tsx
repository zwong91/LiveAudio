import { useState } from 'react';

interface LanguageSelectionProps {
  onLanguageChange: (isSimultaneous: boolean, targetLang: string) => void;
}

const LanguageSelection = ({ onLanguageChange }: LanguageSelectionProps) => {
  const [isSimultaneous, setIsSimultaneous] = useState(false); // 是否启用同声传译
  const [targetLang, setTargetLang] = useState('en'); // 目标语言

  // 处理同声传译启用与禁用
  const handleSimultaneousChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newIsSimultaneous = e.target.checked;
    setIsSimultaneous(newIsSimultaneous);
    onLanguageChange(newIsSimultaneous, targetLang); // 通知父组件更新
  };

  // 处理目标语言更改
  const handleTargetLangChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newTargetLang = e.target.value;
    setTargetLang(newTargetLang);
    onLanguageChange(isSimultaneous, newTargetLang); // 通知父组件更新
  };

  return (
    <div>
      <div>
        <label htmlFor="simultaneous">Enable Simultaneous Translation:</label>
        <input
          type="checkbox"
          id="simultaneous"
          checked={isSimultaneous}
          onChange={handleSimultaneousChange}
        />
      </div>

      <div>
        <label htmlFor="targetLang">Target Language:</label>
        <select
          id="targetLang"
          value={targetLang}
          onChange={handleTargetLangChange}
        >
          <option value="en">English (en)</option>
          <option value="zh">Chinese (zh)</option>
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

      {/* 可选：提供提示信息，当同声传译未启用时，展示额外的提示 */}
      {!isSimultaneous && (
        <p style={{ color: 'gray' }}>Simultaneous translation is not enabled, but you can select a target language chat.</p>
      )}
    </div>
  );
};

export default LanguageSelection;

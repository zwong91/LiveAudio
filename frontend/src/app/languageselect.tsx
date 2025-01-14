import { useState } from 'react';

interface LanguageSelectionProps {
	onLanguageChange: (isSimultaneous: boolean, targetLang: string) => void;
}

const LanguageSelection = ({ onLanguageChange }: LanguageSelectionProps) => {
	const [isSimultaneous, setIsSimultaneous] = useState(false); // 是否启用同声传译
	const [targetLang, setTargetLang] = useState('中文'); // 目标语言

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
				<input type="checkbox" id="simultaneous" checked={isSimultaneous} onChange={handleSimultaneousChange} />
			</div>

			<div>
				<label htmlFor="targetLang">Target Language:</label>
				<select id="targetLang" value={targetLang} onChange={handleTargetLangChange}>
					<option value="中文">Chinese (zh)</option>
					<option value="英语">English (en)</option>
					<option value="韩语">Korean (ko)</option>
					<option value="日语">Japanese (ja)</option>
					<option value="西班牙语">Spanish (es)</option>
					<option value="法语">French (fr)</option>
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

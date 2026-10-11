import os

# المجلدات والملفات التي تريد تجاهلها تماماً (تم إضافة monster_env هنا)
IGNORED_DIRS = {
    '.venv', 'venv', 'env', 'monster_env', 
    '.git', '__pycache__', 'node_modules', 
    '.idea', '.vscode', 'data', 'datasets'
}
IGNORED_FILES = {'bundle_code.py', '.DS_Store', 'package-lock.json'}

# الامتدادات التي تريد فحصها (البايثون فقط لمعرفة الكود الصافي)
ALLOWED_EXTENSIONS = {'.py'}

# الحد الأقصى لحجم الملف المسموح بقراءته (بالكيلوبايت)
MAX_FILE_SIZE_KB = 500 

output_file = "project_source_code.txt"

files_counted = 0
files_skipped_by_size = 0

with open(output_file, 'w', encoding='utf-8') as outfile:
    for root, dirs, files in os.walk('.'):
        # تصفية المجلدات المتجاهلة
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        
        for file in files:
            if file in IGNORED_FILES:
                continue
            
            ext = os.path.splitext(file)[1]
            if ext in ALLOWED_EXTENSIONS:
                file_path = os.path.join(root, file)
                
                try:
                    # حساب حجم الملف الحالي بالكيلوبايت
                    file_size_kb = os.path.getsize(file_path) / 1024
                    
                    # إذا كان الملف أكبر من الحد المسموح، يتم استبعاده فوراً
                    if file_size_kb > MAX_FILE_SIZE_KB:
                        files_skipped_by_size += 1
                        continue
                        
                    outfile.write(f"\n\n{'='*40}\n")
                    outfile.write(f"FILE: {file_path} ({file_size_kb:.2f} KB)\n")
                    outfile.write(f"{'='*40}\n\n")
                    
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                    files_counted += 1
                    
                except Exception as e:
                    outfile.write(f"[Could not read file: {e}]\n")

print(f"\nتم بنجاح! تم جمع كود مشروعك النظيف في ملف: {output_file}")
print(f"📊 إحصائيات سريعة للمشروع:")
print(f"   - عدد ملفات الأكواد والتهيئة التي تم جمعها: {files_counted} ملف.")
print(f"   - عدد ملفات البيانات الضخمة التي تم استبعادها ذكياً: {files_skipped_by_size} ملف.")